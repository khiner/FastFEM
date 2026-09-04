#pragma once

#include "FiniteCell.h"
#include "Tet10Modes.h"
#include "mesh/Tets.h"

#include <expected>
#include <span>
#include <string>

namespace modal {
enum struct Discretization { Tet10,
                             FiniteCell };

struct SurfaceSolveConfig {
    SolverConfig Modal{};
    tetra::Options Tetrahedralization{};
    FiniteCellConfig FiniteCell{};
    float SurfaceSimplificationRatio{1};
};

// Solves a watertight triangle surface with the selected discretization and samples mass-normalized modes at the requested positions.
// Tet10 accepts SolveReuse; finite cell ignores KeepBasis and rejects Tet10 reuse state.
std::expected<ModalResult, std::string> Surface2Modes(std::span<const vec3> positions, std::span<const uint32_t> triangle_indices, const AcousticMaterialProperties &, std::span<const vec3> excitation_positions, vec3 baked_scale, Discretization, SurfaceSolveConfig = {}, SolveReuse = {}, SolveMonitor * = nullptr);
} // namespace modal
