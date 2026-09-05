#pragma once

#include "mesh/TetMesh.h"

#include <expected>
#include <span>
#include <string>

namespace tetra {
struct Options {
    // Inserts interior points until tetrahedra meet a circumradius-to-shortest-edge ratio of 2 where the fixed surface permits refinement.
    // Quality and QualityAndResolution enable this refinement; sliver repair and vertex optimization always run.
    fastfem::TetRefinement Refinement{fastfem::TetRefinement::None};
    // QualityAndResolution also targets this volume in cubed input coordinate units where the fixed surface permits splitting.
    // Required positive and finite for QualityAndResolution; ignored by other modes.
    double MaxVolume{0};
    // Provides one point strictly inside each enclosed void and removes tetrahedra connected to that point without crossing an input face.
    std::span<const dvec3> Holes{};
};

// Wall-clock seconds per stage, with size and effort counters.
struct Profile {
    double DelaunaySeconds{}, RecoverSeconds{}, CarveSeconds{}, RefineSeconds{};
    // SegmentSeconds, FaceSeconds, and SuppressSeconds partition the measured recovery passes.
    // RecoverSeconds also includes constraint setup and surface marking.
    double SegmentSeconds{}, FaceSeconds{}, SuppressSeconds{};
    // Reports sizes from the returned build.
    uint32_t TetCount{}, SteinerCount{};
    // Counts interior tetrahedra before surface recovery.
    uint32_t DelaunayTetCount{};
    // Counts retained boundary Steiner points and replacement interior points.
    uint32_t BdrySteinerCount{}, VolSteinerCount{};
    uint32_t FlipCount{}, SplitCount{}, MissingEdgeCount{}, MissingFaceCount{}, Builds{};
};

struct Result {
    TetMesh Mesh;
    Profile Profile;
};

// Returns a constrained Delaunay tetrahedralization of a closed, non-self-intersecting triangle surface.
// `triangle_indices` contains three indices per face.
// Input edges may have more than two incident triangles, and triangle orientation may vary.
// The result preserves every input vertex index and input triangle and places every Steiner point inside the surface.
// Invalid or unrecoverable surfaces return an error string.
std::expected<Result, std::string> Tetrahedralize(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices, Options options = {});
} // namespace tetra
