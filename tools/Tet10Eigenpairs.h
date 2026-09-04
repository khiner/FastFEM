#pragma once

#include "audio/mesh2modes.h"
#include "mesh/TetMesh.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <numbers>

struct Tet10Eigenpairs {
    Eigen::VectorXd Eigenvalues;
    Eigen::MatrixXd Eigenvectors;
    double RelativeResidual{};
};

inline Tet10Eigenpairs SolveTet10Eigenpairs(
    const TetMesh &mesh, const AcousticMaterialProperties &material, uint32_t count,
    double alpha, double tolerance, uint32_t max_restarts
) {
    modal::SolveCache cache;
    const auto result = modal::mesh2modes(
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
    return {
        .Eigenvalues = Eigen::Map<const Eigen::VectorXd>{
            result.Summary.Eigenvalues.data(), Eigen::Index(result.Summary.Eigenvalues.size())
        },
        .Eigenvectors = result.Basis.cast<double>(),
        .RelativeResidual = result.Profile.PhysicalResidual,
    };
}
