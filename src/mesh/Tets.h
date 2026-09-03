#pragma once

#include "mesh/Tetrahedralize.h"

// Simplifies the surface in place to `ratio` of its triangles with quadric edge collapse and removes unreferenced vertices.
// A ratio greater than or equal to one preserves the input.
// The function retries folds with their neighborhoods fixed and preserves resolution where every collapse causes an intersection.
void SimplifySurface(std::vector<vec3> &positions, std::vector<uint32_t> &triangle_indices, float ratio);

// Returns a tetrahedral mesh whose boundary contains the closed input triangle surface exactly.
// Returns an error string for invalid or unrecoverable input.
// Call SimplifySurface first to reduce surface resolution.
std::expected<tetra::Result, std::string> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, tetra::Options options = {});
