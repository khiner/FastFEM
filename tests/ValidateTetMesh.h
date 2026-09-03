#pragma once

// Validates vertex preservation, tetrahedron orientation, face incidence, boundary coverage, input-triangle coverage, and enclosed volume.
// Returns the first structural defect or an empty string for a valid mesh.

#include "mesh/TetMesh.h"
#include "numeric/Predicates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>

// Returns true when p lies inside triangle (a, b, c) within a scaled coplanarity tolerance.
inline bool OnTriangle(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &p) {
    const dvec3 n = numeric::Cross(b - a, c - a);
    const double n2 = numeric::Dot(n, n);
    if (n2 == 0) return false;
    const double scale = std::sqrt(numeric::Dot(b - a, b - a) * numeric::Dot(c - a, c - a));
    const double dist = numeric::Dot(n, p - a) / std::sqrt(n2);
    if (std::abs(dist) > 1e-9 * std::sqrt(scale)) return false;
    const dvec3 corners[3]{a, b, c};
    for (int e = 0; e < 3; ++e) {
        const dvec3 &u = corners[e], &v = corners[(e + 1) % 3], &w = corners[(e + 2) % 3];
        const dvec3 inward = numeric::Cross(n, v - u);
        const double side_p = numeric::Dot(inward, p - u), side_w = numeric::Dot(inward, w - u);
        if (side_w == 0 || side_p / side_w < -1e-9) return false;
    }
    return true;
}

// Returns three sorted vertex indices for winding-independent triangle comparison.
inline std::array<uint32_t, 3> SortedTri(uint32_t a, uint32_t b, uint32_t c) {
    std::array tri{a, b, c};
    std::ranges::sort(tri);
    return tri;
}

inline std::string ValidateTetMesh(std::span<const dvec3> in_points, std::span<const uint32_t> in_tris, const TetMesh &mesh) {
    const uint32_t n = uint32_t(in_points.size());
    for (uint32_t i = 0; i < n; ++i) {
        if (mesh.Points[i] != in_points[i]) return "input vertex moved";
    }
    // Sorting each face and the face list groups all tetrahedra incident to one face.
    std::vector<std::array<uint32_t, 3>> faces;
    faces.reserve(mesh.Tets.size() * 4);
    for (const auto &t : mesh.Tets) {
        if (geom::Orient3D(mesh.Points[t[0]], mesh.Points[t[1]], mesh.Points[t[2]], mesh.Points[t[3]]) <= 0) return "non-positive tet";
        constexpr uint32_t fc[4][3]{{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
        for (const auto &f : fc) faces.push_back(SortedTri(t[f[0]], t[f[1]], t[f[2]]));
    }
    std::ranges::sort(faces);
    // Duplicate input triangles represent zero-thickness internal flaps rather than tetrahedral boundaries.
    std::vector<std::array<uint32_t, 3>> input_faces, flaps;
    input_faces.reserve(in_tris.size() / 3);
    for (size_t i = 0; i < in_tris.size(); i += 3) input_faces.push_back(SortedTri(in_tris[i], in_tris[i + 1], in_tris[i + 2]));
    std::ranges::sort(input_faces);
    for (size_t i = 1; i < input_faces.size(); ++i) {
        if (input_faces[i] == input_faces[i - 1] && (flaps.empty() || flaps.back() != input_faces[i])) flaps.push_back(input_faces[i]);
    }
    input_faces.erase(std::ranges::unique(input_faces).begin(), input_faces.end());
    const auto tri_within = [&](const dvec3 &a, const dvec3 &b, const dvec3 &c, const std::array<uint32_t, 3> &tri) {
        return std::ranges::all_of(tri, [&](uint32_t v) { return OnTriangle(a, b, c, mesh.Points[v]); });
    };
    const auto inside_some_input = [&](const std::array<uint32_t, 3> &tri) {
        return std::ranges::any_of(input_faces, [&](const auto &it) { return tri_within(in_points[it[0]], in_points[it[1]], in_points[it[2]], tri); });
    };
    // A face with one incident tetrahedron belongs to the mesh boundary.
    std::vector<std::array<uint32_t, 3>> boundary;
    for (size_t i = 0; i < faces.size();) {
        size_t j = i;
        while (j < faces.size() && faces[j] == faces[i]) ++j;
        if (j - i == 1) {
            if (!std::ranges::binary_search(input_faces, faces[i]) && !inside_some_input(faces[i])) return "boundary face is not on the input surface";
            boundary.push_back(faces[i]);
        } else if (j - i != 2) {
            return "face shared by more than two tets";
        }
        i = j;
    }
    // An internal input triangle may have tetrahedra on both sides.
    // Exact face presence or a contained refined boundary face satisfies input-triangle coverage.
    for (const auto &tri : input_faces) {
        if (std::ranges::binary_search(faces, tri) || std::ranges::binary_search(flaps, tri)) continue;
        const dvec3 &a = in_points[tri[0]], &b = in_points[tri[1]], &c = in_points[tri[2]];
        if (!std::ranges::any_of(boundary, [&](const auto &mtri) { return tri_within(a, b, c, mtri); })) return "input triangle missing from tet mesh";
    }
    // Face incidence verifies topology but cannot detect geometric overlap or gaps between tetrahedra.
    // Equal surface and tetrahedron volumes detect those defects for consistently wound watertight input.
    // Non-manifold input lacks a unique divergence-sum volume and therefore uses the remaining structural checks.
    std::vector<std::pair<uint64_t, int>> edges;
    edges.reserve(in_tris.size());
    for (size_t i = 0; i < in_tris.size(); i += 3)
        for (int e = 0; e < 3; ++e) {
            const uint32_t u = in_tris[i + e], v = in_tris[i + (e + 1) % 3];
            edges.emplace_back((uint64_t(std::min(u, v)) << 32) | std::max(u, v), u < v ? 1 : -1);
        }
    std::ranges::sort(edges);
    bool clean_manifold = true;
    for (size_t i = 0; i < edges.size() && clean_manifold;) {
        size_t j = i;
        int dir = 0;
        while (j < edges.size() && edges[j].first == edges[i].first) dir += edges[j++].second;
        if (j - i != 2 || dir != 0) clean_manifold = false;
        i = j;
    }
    if (clean_manifold) {
        // Both determinant sums equal six times signed enclosed volume.
        // Absolute values account for surface winding, and coplanar refinements preserve the surface sum.
        double mesh_vol = 0;
        for (const auto &t : mesh.Tets) {
            const dvec3 &a = mesh.Points[t[0]];
            mesh_vol += numeric::Dot(mesh.Points[t[1]] - a, numeric::Cross(mesh.Points[t[2]] - a, mesh.Points[t[3]] - a));
        }
        double surf_vol = 0;
        for (size_t i = 0; i < in_tris.size(); i += 3)
            surf_vol += numeric::Dot(in_points[in_tris[i]], numeric::Cross(in_points[in_tris[i + 1]], in_points[in_tris[i + 2]]));
        if (std::abs(std::abs(mesh_vol) - std::abs(surf_vol)) > 1e-6 * std::abs(surf_vol)) return "mesh volume does not match the surface";
    }
    return {};
}
