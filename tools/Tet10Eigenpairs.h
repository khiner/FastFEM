#pragma once

#include "audio/Tet10Modes.h"
#include "mesh/TetMesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

struct Tet10Eigenpairs {
    numeric::Vector<double> Eigenvalues;
    numeric::Matrix<double> Eigenvectors;
    double RelativeResidual{};
};

inline Tet10Eigenpairs SolveTet10Eigenpairs(
    const TetMesh &mesh, const AcousticMaterialProperties &material, uint32_t count,
    double alpha, double tolerance, uint32_t max_restarts
) {
    modal::SolveCache cache;
    const auto result = modal::SolveTet10Modes(
        mesh, material, {vec3(mesh.Points.front())}, vec3{1},
        {
            .MinModeFreq = float(std::sqrt(alpha) / (2 * std::numbers::pi)),
            .MaxModeFreq = std::numeric_limits<float>::max(),
            .NumModes = count,
            .NumFemModes = count,
            .Tolerance = tolerance,
            .MaxRestarts = max_restarts,
        },
        {.Cache = &cache, .KeepBasis = true}
    );
    numeric::Vector<double> eigenvalues(result.Summary.Eigenvalues.size());
    std::ranges::copy(result.Summary.Eigenvalues, eigenvalues.begin());
    return {
        .Eigenvalues = std::move(eigenvalues),
        .Eigenvectors = numeric::Cast<double>(result.Basis.View()),
        .RelativeResidual = result.Profile.PhysicalResidual,
    };
}
