#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <vector>

// Stores raw eigenpairs sampled at excitation positions for modal-model rescaling.
// RescaleModes derives exact modes after density or Young's modulus changes by using the solved material and unchanged FEM inputs.
struct ModalEigenSummary {
    std::vector<double> Eigenvalues; // ascending, all solved eigenpairs
    std::vector<std::vector<vec3>> Shapes; // mass-normalized, by [excitation position][eigenpair]
    AcousticMaterialProperties SolvedMaterial{};

    bool operator==(const ModalEigenSummary &) const = default;
};
