#pragma once

#include "numeric/vec3.h"

#include <array>
#include <cstdint>
#include <vector>

// Stores a tetrahedral volume mesh.
// Every tetrahedron (a, b, c, d) satisfies det[a-d, b-d, c-d] > 0.
struct TetMesh {
    std::vector<dvec3> Points;
    std::vector<std::array<uint32_t, 4>> Tets;
};
