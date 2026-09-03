#pragma once

#include "FiniteCell.h"
#include "mesh/Tets.h"
#include "mesh2modes.h"

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
};

// Solves a watertight triangle surface with the selected discretization and samples mass-normalized modes at the requested positions.
// Tet10 accepts every SolveReuse field, while finite cell accepts KeepBasis.
std::expected<ModalResult, std::string> surface2modes(std::span<const vec3> positions, std::span<const uint32_t> triangle_indices, const AcousticMaterialProperties &, std::span<const vec3> excitation_positions, vec3 baked_scale, Discretization, SurfaceSolveConfig = {}, SolveReuse = {}, SolveMonitor * = nullptr);
} // namespace modal
