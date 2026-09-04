#pragma once

#include "StructuredBar.h"
#include "mesh/Tetrahedralize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace finite_cell_benchmark {
struct Surface {
    std::vector<dvec3> Points;
    std::vector<uint32_t> Triangles;
};

struct Geometry {
    std::string_view Name;
    std::string_view Description;
    Surface Boundary;
    dvec3 InteriorPoint;
    dvec3 GridExtent;
    dvec3 TetExtent;
    double PhysicalVolume;
    TetMesh (*BuildTetMesh)(uint32_t, uint32_t, uint32_t);
};

inline dvec3 Rotate(dvec3 point, dvec3 center, double az, double ay) {
    point -= center;
    point = {
        std::cos(ay) * (std::cos(az) * point.x - std::sin(az) * point.y) + std::sin(ay) * point.z,
        std::sin(az) * point.x + std::cos(az) * point.y,
        -std::sin(ay) * (std::cos(az) * point.x - std::sin(az) * point.y) + std::cos(ay) * point.z,
    };
    return point + center;
}

inline constexpr std::array<uint32_t, 36> BoxTriangles{
    0,
    2,
    3,
    0,
    3,
    1,
    4,
    5,
    7,
    4,
    7,
    6,
    0,
    1,
    5,
    0,
    5,
    4,
    2,
    6,
    7,
    2,
    7,
    3,
    0,
    4,
    6,
    0,
    6,
    2,
    1,
    3,
    7,
    1,
    7,
    5,
};

inline Surface BoxSurface(dvec3 extent, double az, double ay) {
    Surface surface{
        .Points = {
            {0, 0, 0},
            {extent.x, 0, 0},
            {0, extent.y, 0},
            {extent.x, extent.y, 0},
            {0, 0, extent.z},
            {extent.x, 0, extent.z},
            {0, extent.y, extent.z},
            extent,
        },
        .Triangles = {BoxTriangles.begin(), BoxTriangles.end()},
    };
    const dvec3 center = 0.5 * extent;
    for (auto &point : surface.Points) point = Rotate(point, center, az, ay);
    return surface;
}

inline constexpr dvec3 BarExtent{0.3, 0.075, 0.05};

inline Surface AxisBarSurface() { return BoxSurface(BarExtent, 0, 0); }
inline Surface RotatedBarSurface() { return BoxSurface(BarExtent, 0.23, -0.14); }
inline Surface SteepBarSurface() { return BoxSurface(BarExtent, 0.61, 0.37); }

inline TetMesh BuildBarTetMesh(uint32_t nx, uint32_t ny, uint32_t nz, double az, double ay) {
    constexpr dvec3 extent = BarExtent;
    TetMesh mesh = MakeStructuredBar(int(nx), int(ny), int(nz), extent);
    const dvec3 center = 0.5 * extent;
    for (auto &point : mesh.Points) {
        point = Rotate(point, center, az, ay);
    }
    return mesh;
}

inline TetMesh BuildAxisBarTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return BuildBarTetMesh(nx, ny, nz, 0, 0); }
inline TetMesh BuildRotatedBarTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return BuildBarTetMesh(nx, ny, nz, 0.23, -0.14); }
inline TetMesh BuildSteepBarTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return BuildBarTetMesh(nx, ny, nz, 0.61, 0.37); }

inline Surface Icosphere(uint32_t subdivisions) {
    constexpr double phi = std::numbers::phi;
    std::vector<dvec3> points{
        {-1, phi, 0},
        {1, phi, 0},
        {-1, -phi, 0},
        {1, -phi, 0},
        {0, -1, phi},
        {0, 1, phi},
        {0, -1, -phi},
        {0, 1, -phi},
        {phi, 0, -1},
        {phi, 0, 1},
        {-phi, 0, -1},
        {-phi, 0, 1},
    };
    std::vector<std::array<uint32_t, 3>> triangles{
        {0, 11, 5},
        {0, 5, 1},
        {0, 1, 7},
        {0, 7, 10},
        {0, 10, 11},
        {1, 5, 9},
        {5, 11, 4},
        {11, 10, 2},
        {10, 7, 6},
        {7, 1, 8},
        {3, 9, 4},
        {3, 4, 2},
        {3, 2, 6},
        {3, 6, 8},
        {3, 8, 9},
        {4, 9, 5},
        {2, 4, 11},
        {6, 2, 10},
        {8, 6, 7},
        {9, 8, 1},
    };
    for (auto &point : points) point = numeric::Normalize(point);
    for (uint32_t subdivision = 0; subdivision < subdivisions; ++subdivision) {
        std::map<uint64_t, uint32_t> midpoints;
        const auto Midpoint = [&](uint32_t a, uint32_t b) {
            const uint64_t key = uint64_t(std::min(a, b)) << 32 | std::max(a, b);
            const auto [entry, inserted] = midpoints.try_emplace(key, uint32_t(points.size()));
            if (inserted) points.push_back(numeric::Normalize(0.5 * (points[a] + points[b])));
            return entry->second;
        };
        std::vector<std::array<uint32_t, 3>> refined;
        refined.reserve(4 * triangles.size());
        for (const auto &triangle : triangles) {
            const uint32_t ab = Midpoint(triangle[0], triangle[1]);
            const uint32_t bc = Midpoint(triangle[1], triangle[2]);
            const uint32_t ca = Midpoint(triangle[2], triangle[0]);
            refined.insert(refined.end(), {{triangle[0], ab, ca}, {triangle[1], bc, ab}, {triangle[2], ca, bc}, {ab, bc, ca}});
        }
        triangles = std::move(refined);
    }
    Surface surface{.Points = std::move(points)};
    surface.Triangles.reserve(3 * triangles.size());
    for (const auto &triangle : triangles)
        surface.Triangles.insert(surface.Triangles.end(), triangle.begin(), triangle.end());
    return surface;
}

inline Surface EllipsoidSurface(uint32_t subdivisions) {
    constexpr dvec3 center{0.15, 0.09, 0.065}, radii{0.15, 0.09, 0.065};
    auto surface = Icosphere(subdivisions);
    for (auto &point : surface.Points) {
        point = center + radii * point;
        point = Rotate(point, center, 0.31, -0.19);
    }
    return surface;
}

inline Surface LBracketSurface() {
    constexpr double length = 0.24, arm = 0.08, thickness = 0.045;
    constexpr std::array<dvec3, 6> polygon{
        dvec3{0, 0, 0},
        {length, 0, 0},
        {length, arm, 0},
        {arm, arm, 0},
        {arm, length, 0},
        {0, length, 0},
    };
    Surface surface{
        .Points = {polygon.begin(), polygon.end()},
        .Triangles = {0, 1, 3, 1, 2, 3, 0, 3, 5, 3, 4, 5, 6, 9, 7, 7, 9, 8, 6, 11, 9, 9, 11, 10},
    };
    surface.Points.reserve(12);
    for (auto point : polygon) {
        point.z = thickness;
        surface.Points.push_back(point);
    }
    for (uint32_t edge = 0; edge < polygon.size(); ++edge) {
        const uint32_t next = (edge + 1) % polygon.size();
        surface.Triangles.insert(surface.Triangles.end(), {edge, next, next + 6, edge, next + 6, edge + 6});
    }
    constexpr dvec3 center{0.12, 0.12, 0.0225};
    for (auto &point : surface.Points) point = Rotate(point, center, -0.24, 0.16);
    return surface;
}

inline Surface RevolvedSolidSurface(const std::vector<double> &axial, const std::vector<double> &radii) {
    constexpr uint32_t segments{48};
    if (axial.size() != radii.size() || axial.size() < 2)
        throw std::invalid_argument("A revolved solid requires matching axial and radial profiles.");
    Surface surface;
    surface.Points.reserve(axial.size() * segments + 2);
    for (uint32_t station = 0; station < axial.size(); ++station)
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const double angle = 2 * std::numbers::pi * segment / segments;
            surface.Points.push_back({radii[station] * std::cos(angle), radii[station] * std::sin(angle), axial[station]});
        }
    for (uint32_t station = 0; station + 1 < axial.size(); ++station)
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const uint32_t next = (segment + 1) % segments;
            const uint32_t a = station * segments + segment, b = station * segments + next;
            const uint32_t c = (station + 1) * segments + segment, d = (station + 1) * segments + next;
            surface.Triangles.insert(surface.Triangles.end(), {a, b, d, a, d, c});
        }
    const uint32_t bottom = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, axial.front()});
    const uint32_t top = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, axial.back()});
    const uint32_t top_offset = uint32_t(axial.size() - 1) * segments;
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t next = (segment + 1) % segments;
        surface.Triangles.insert(surface.Triangles.end(), {bottom, next, segment, top, top_offset + segment, top_offset + next});
    }
    return surface;
}

inline Surface ThinPlateSurface() {
    constexpr dvec3 extent{0.24, 0.18, 0.008};
    return BoxSurface(extent, 0.17, -0.11);
}

inline Surface DiscSurface() {
    auto surface = RevolvedSolidSurface({0, 0.008}, {0.12, 0.12});
    constexpr dvec3 center{0, 0, 0.004};
    for (auto &point : surface.Points) point = Rotate(point, center, 0.13, -0.09);
    return surface;
}

inline Surface TorusSurface() {
    constexpr uint32_t major_segments{48}, minor_segments{16};
    constexpr double major_radius = 0.09, minor_radius = 0.018;
    Surface surface;
    surface.Points.reserve(major_segments * minor_segments);
    for (uint32_t major = 0; major < major_segments; ++major) {
        const double a = 2 * std::numbers::pi * major / major_segments;
        for (uint32_t minor = 0; minor < minor_segments; ++minor) {
            const double b = 2 * std::numbers::pi * minor / minor_segments;
            const double radius = major_radius + minor_radius * std::cos(b);
            surface.Points.push_back({radius * std::cos(a), radius * std::sin(a), minor_radius * std::sin(b)});
        }
    }
    const auto Node = [=](uint32_t major, uint32_t minor) {
        return (major % major_segments) * minor_segments + minor % minor_segments;
    };
    for (uint32_t major = 0; major < major_segments; ++major)
        for (uint32_t minor = 0; minor < minor_segments; ++minor) {
            const uint32_t a = Node(major, minor), b = Node(major + 1, minor);
            const uint32_t c = Node(major, minor + 1), d = Node(major + 1, minor + 1);
            surface.Triangles.insert(surface.Triangles.end(), {a, b, d, a, d, c});
        }
    for (auto &point : surface.Points) point = Rotate(point, {}, 0.21, -0.16);
    return surface;
}

inline Surface TaperedKeySurface() {
    constexpr double length = 0.25, root_width = 0.032, root_height = 0.009, tip_width = 0.014, tip_height = 0.004;
    Surface surface{
        .Points = {
            {0, -root_width, -root_height},
            {length, -tip_width, -tip_height},
            {0, root_width, -root_height},
            {length, tip_width, -tip_height},
            {0, -root_width, root_height},
            {length, -tip_width, tip_height},
            {0, root_width, root_height},
            {length, tip_width, tip_height},
        },
        .Triangles = {BoxTriangles.begin(), BoxTriangles.end()},
    };
    constexpr dvec3 center{0.125, 0, 0};
    for (auto &point : surface.Points) point = Rotate(point, center, -0.19, 0.12);
    return surface;
}

inline Surface NarrowWaistSurface() {
    auto surface = RevolvedSolidSurface({0, 0.04, 0.10, 0.14, 0.20, 0.24}, {0.052, 0.064, 0.026, 0.026, 0.064, 0.052});
    constexpr dvec3 center{0, 0, 0.12};
    for (auto &point : surface.Points) point = Rotate(point, center, 0.24, -0.13);
    return surface;
}

inline Surface CupSurface() {
    constexpr uint32_t segments{48}, stations{8};
    constexpr double height = 0.14, bottom = 0.018, outer_bottom = 0.042, outer_top = 0.10, wall = 0.008;
    Surface surface;
    surface.Points.reserve(2 * stations * segments + 2);
    const auto OuterRadius = [=](double t) { return outer_bottom + (outer_top - outer_bottom) * (0.25 * t + 0.75 * t * t); };
    for (uint32_t station = 0; station < stations; ++station) {
        const double t = double(station) / (stations - 1), z = height * t, radius = OuterRadius(t);
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const double angle = 2 * std::numbers::pi * segment / segments;
            surface.Points.push_back({radius * std::cos(angle), radius * std::sin(angle), z});
        }
    }
    const uint32_t inner_offset = uint32_t(surface.Points.size());
    for (uint32_t station = 0; station < stations; ++station) {
        const double t = double(station) / (stations - 1), z = bottom + (height - bottom) * t;
        const double outer_t = z / height, radius = OuterRadius(outer_t) - wall;
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const double angle = 2 * std::numbers::pi * segment / segments;
            surface.Points.push_back({radius * std::cos(angle), radius * std::sin(angle), z});
        }
    }
    for (uint32_t station = 0; station + 1 < stations; ++station)
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const uint32_t next = (segment + 1) % segments;
            const uint32_t a = station * segments + segment, b = station * segments + next;
            const uint32_t c = (station + 1) * segments + segment, d = (station + 1) * segments + next;
            surface.Triangles.insert(surface.Triangles.end(), {a, b, d, a, d, c});
            const uint32_t ia = inner_offset + a, ib = inner_offset + b, ic = inner_offset + c, id = inner_offset + d;
            surface.Triangles.insert(surface.Triangles.end(), {ia, ic, id, ia, id, ib});
        }
    const uint32_t bottom_center = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, 0});
    const uint32_t cavity_center = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, bottom});
    const uint32_t outer_top_offset = (stations - 1) * segments, inner_top_offset = inner_offset + outer_top_offset;
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t next = (segment + 1) % segments;
        surface.Triangles.insert(surface.Triangles.end(), {bottom_center, next, segment});
        surface.Triangles.insert(surface.Triangles.end(), {cavity_center, inner_offset + segment, inner_offset + next});
        surface.Triangles.insert(surface.Triangles.end(), {outer_top_offset + segment, outer_top_offset + next, inner_top_offset + next, outer_top_offset + segment, inner_top_offset + next, inner_top_offset + segment});
    }
    constexpr dvec3 center{0, 0, height / 2};
    for (auto &point : surface.Points) point = Rotate(point, center, -0.17, 0.10);
    return surface;
}

inline std::pair<dvec3, dvec3> Bounds(const Surface &surface) {
    dvec3 min{std::numeric_limits<double>::infinity()}, max{-std::numeric_limits<double>::infinity()};
    for (const auto &point : surface.Points) {
        min = numeric::Min(min, point);
        max = numeric::Max(max, point);
    }
    return {min, max};
}

inline double Volume(const Surface &surface) {
    double volume{};
    for (size_t triangle = 0; triangle < surface.Triangles.size(); triangle += 3) {
        const auto &a = surface.Points[surface.Triangles[triangle]];
        const auto &b = surface.Points[surface.Triangles[triangle + 1]];
        const auto &c = surface.Points[surface.Triangles[triangle + 2]];
        volume += numeric::Dot(a, numeric::Cross(b, c)) / 6;
    }
    return std::abs(volume);
}

inline TetMesh TetrahedralizeSurface(
    const Surface &surface, uint32_t nx, uint32_t ny, uint32_t nz, std::span<const dvec3> holes = {}
) {
    const auto [min, max] = Bounds(surface);
    const double max_volume = ((max.x - min.x) * (max.y - min.y) * (max.z - min.z)) / (6.0 * nx * ny * nz);
    auto result = tetra::Tetrahedralize(surface.Points, surface.Triangles, {.MaxVolume = max_volume, .Holes = holes});
    if (!result) throw std::runtime_error("Surface tetrahedralization failed: " + result.error());
    return std::move(result->Mesh);
}

inline Surface CoarseEllipsoidSurface() { return EllipsoidSurface(2); }
inline Surface DenseEllipsoidSurface() { return EllipsoidSurface(4); }

inline TetMesh BuildEllipsoidTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) {
    return TetrahedralizeSurface(CoarseEllipsoidSurface(), nx, ny, nz);
}

inline TetMesh BuildDenseEllipsoidTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) {
    return TetrahedralizeSurface(DenseEllipsoidSurface(), nx, ny, nz);
}

inline TetMesh BuildLBracketTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) {
    constexpr double length = 0.24, arm = 0.08, thickness = 0.045;
    const uint32_t bx = std::clamp(uint32_t(std::lround(nx * arm / length)), 1u, nx - 1);
    const uint32_t by = std::clamp(uint32_t(std::lround(ny * arm / length)), 1u, ny - 1);
    const auto Coordinate = [=](uint32_t i, uint32_t count, uint32_t split, double end) {
        return i <= split ? arm * i / split : arm + (end - arm) * (i - split) / (count - split);
    };
    TetMesh mesh;
    std::map<std::array<uint32_t, 3>, uint32_t> nodes;
    const auto Node = [&](uint32_t x, uint32_t y, uint32_t z) {
        const std::array key{x, y, z};
        const auto [entry, inserted] = nodes.try_emplace(key, uint32_t(mesh.Points.size()));
        if (inserted)
            mesh.Points.push_back({Coordinate(x, nx, bx, length), Coordinate(y, ny, by, length), thickness * z / nz});
        return entry->second;
    };
    for (uint32_t x = 0; x < nx; ++x)
        for (uint32_t y = 0; y < ny; ++y) {
            if (x >= bx && y >= by) continue;
            for (uint32_t z = 0; z < nz; ++z) {
                const uint32_t cell[8]{
                    Node(x, y, z),
                    Node(x + 1, y, z),
                    Node(x, y + 1, z),
                    Node(x + 1, y + 1, z),
                    Node(x, y, z + 1),
                    Node(x + 1, y, z + 1),
                    Node(x, y + 1, z + 1),
                    Node(x + 1, y + 1, z + 1),
                };
                for (const auto &tet : HexTets) mesh.Tets.push_back({cell[tet[0]], cell[tet[1]], cell[tet[2]], cell[tet[3]]});
            }
        }
    constexpr dvec3 center{0.12, 0.12, 0.0225};
    for (auto &point : mesh.Points) point = Rotate(point, center, -0.24, 0.16);
    return mesh;
}

inline TetMesh BuildThinPlateTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(ThinPlateSurface(), nx, ny, nz); }
inline TetMesh BuildDiscTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(DiscSurface(), nx, ny, nz); }
inline TetMesh BuildTorusTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(TorusSurface(), nx, ny, nz); }
inline TetMesh BuildTaperedKeyTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(TaperedKeySurface(), nx, ny, nz); }
inline TetMesh BuildNarrowWaistTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(NarrowWaistSurface(), nx, ny, nz); }
inline TetMesh BuildCupTetMesh(uint32_t nx, uint32_t ny, uint32_t nz) { return TetrahedralizeSurface(CupSurface(), nx, ny, nz); }

// Derives grid extent and physical volume from the boundary surface unless an entry overrides them.
// The `bar` preserves an axis-aligned grid, while the polyhedral bars and bracket use exact analytical volumes.
struct GeometrySpec {
    std::string_view Name, Description;
    Surface (*Boundary)();
    dvec3 InteriorPoint, TetExtent;
    TetMesh (*BuildTetMesh)(uint32_t, uint32_t, uint32_t);
    dvec3 GridExtent{};
    double PhysicalVolume{};
};

inline Geometry MakeGeometry(std::string_view name) {
    static const std::array specs{
        GeometrySpec{"bar-axis", "axis-aligned rectangular bar", AxisBarSurface, 0.5 * BarExtent, BarExtent, BuildAxisBarTetMesh, BarExtent, BarExtent.x * BarExtent.y * BarExtent.z},
        GeometrySpec{"bar", "historical rotated rectangular bar", RotatedBarSurface, 0.5 * BarExtent, BarExtent, BuildRotatedBarTetMesh, BarExtent, BarExtent.x * BarExtent.y * BarExtent.z},
        GeometrySpec{"bar-rotated", "rotated rectangular bar", RotatedBarSurface, 0.5 * BarExtent, BarExtent, BuildRotatedBarTetMesh, {}, BarExtent.x * BarExtent.y * BarExtent.z},
        GeometrySpec{"bar-steep", "steeply rotated rectangular bar", SteepBarSurface, 0.5 * BarExtent, BarExtent, BuildSteepBarTetMesh, {}, BarExtent.x * BarExtent.y * BarExtent.z},
        GeometrySpec{"ellipsoid", "curved ellipsoid", CoarseEllipsoidSurface, {0.15, 0.09, 0.065}, {0.3, 0.18, 0.13}, BuildEllipsoidTetMesh},
        GeometrySpec{"ellipsoid-dense", "dense curved ellipsoid", DenseEllipsoidSurface, {0.15, 0.09, 0.065}, {0.3, 0.18, 0.13}, BuildDenseEllipsoidTetMesh},
        GeometrySpec{"l-bracket", "non-convex L bracket", LBracketSurface, Rotate({0.04, 0.12, 0.0225}, {0.12, 0.12, 0.0225}, -0.24, 0.16), {0.24, 0.24, 0.045}, BuildLBracketTetMesh, {}, (2 * 0.24 * 0.08 - 0.08 * 0.08) * 0.045},
        GeometrySpec{"thin-plate", "thin rectangular plate", ThinPlateSurface, Rotate({0.12, 0.09, 0.004}, {0.12, 0.09, 0.004}, 0.17, -0.11), {0.24, 0.18, 0.008}, BuildThinPlateTetMesh},
        GeometrySpec{"disc", "thin circular disc", DiscSurface, Rotate({0, 0, 0.004}, {0, 0, 0.004}, 0.13, -0.09), {0.24, 0.24, 0.008}, BuildDiscTetMesh},
        GeometrySpec{"torus", "solid resonant ring", TorusSurface, Rotate({0.09, 0, 0}, {}, 0.21, -0.16), {0.216, 0.216, 0.036}, BuildTorusTetMesh},
        GeometrySpec{"tapered-key", "tapered percussion key", TaperedKeySurface, {0.125, 0, 0}, {0.25, 0.064, 0.018}, BuildTaperedKeyTetMesh},
        GeometrySpec{"narrow-waist", "weakly coupled narrow-waist resonator", NarrowWaistSurface, Rotate({0, 0, 0.12}, {0, 0, 0.12}, 0.24, -0.13), {0.128, 0.128, 0.24}, BuildNarrowWaistTetMesh},
        GeometrySpec{"cup", "hollow cup resonator", CupSurface, Rotate({0, 0, 0.006}, {0, 0, 0.07}, -0.17, 0.10), {0.2, 0.2, 0.14}, BuildCupTetMesh},
    };
    const auto spec = std::ranges::find(specs, name, &GeometrySpec::Name);
    if (spec == specs.end()) throw std::invalid_argument("Unknown finite-cell geometry: " + std::string{name});
    Geometry result{
        .Name = spec->Name,
        .Description = spec->Description,
        .Boundary = spec->Boundary(),
        .InteriorPoint = spec->InteriorPoint,
        .GridExtent = spec->GridExtent,
        .TetExtent = spec->TetExtent,
        .PhysicalVolume = spec->PhysicalVolume,
        .BuildTetMesh = spec->BuildTetMesh,
    };
    const auto [min, max] = Bounds(result.Boundary);
    if (result.GridExtent == dvec3{}) result.GridExtent = max - min;
    if (result.PhysicalVolume == 0) result.PhysicalVolume = Volume(result.Boundary);
    return result;
}

inline constexpr std::array<std::string_view, 6> AudioGeometryNames{
    "thin-plate",
    "disc",
    "torus",
    "tapered-key",
    "narrow-waist",
    "cup",
};
inline constexpr std::array<std::string_view, 13> GeometryNames{
    "bar",
    "bar-axis",
    "bar-rotated",
    "bar-steep",
    "ellipsoid",
    "ellipsoid-dense",
    "l-bracket",
    "thin-plate",
    "disc",
    "torus",
    "tapered-key",
    "narrow-waist",
    "cup",
};
inline bool IsGeometry(std::string_view name) { return std::ranges::find(GeometryNames, name) != GeometryNames.end(); }

inline uvec3 Resolution(dvec3 extent, uint32_t longitudinal) {
    const double longest = std::max({extent.x, extent.y, extent.z});
    return {
        std::max(2u, uint32_t(std::lround(longitudinal * extent.x / longest))),
        std::max(2u, uint32_t(std::lround(longitudinal * extent.y / longest))),
        std::max(2u, uint32_t(std::lround(longitudinal * extent.z / longest))),
    };
}

inline uvec3 GridResolution(const Geometry &geometry, uint32_t longitudinal) {
    return Resolution(geometry.GridExtent, longitudinal);
}

inline uvec3 TetResolution(const Geometry &geometry, uint32_t longitudinal) {
    return Resolution(geometry.TetExtent, longitudinal);
}
} // namespace finite_cell_benchmark
