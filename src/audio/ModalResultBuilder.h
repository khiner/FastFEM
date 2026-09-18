#pragma once

#include "Tet10Modes.h"

namespace modal {
using numeric::vec3;

ModalResult BuildModalResult(std::vector<double> eigenvalues, std::vector<std::vector<vec3>> shapes, const AcousticMaterialProperties &, const SolverConfig &, std::vector<vec3> positions, vec3 baked_scale, MassProperties, SolveProfile, Matrix<float> basis, std::vector<uint32_t> sample_point_of);
} // namespace modal
