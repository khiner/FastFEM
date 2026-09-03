#pragma once

#include "audio/CholeskyShiftInvert.h"
#include "audio/FiniteCell.h"
#include "audio/GeneralizedEigenSolver.h"

#include <algorithm>

namespace finite_cell_benchmark {
struct ReferenceModes {
    Eigen::VectorXd Values;
    Eigen::MatrixXd Vectors;
    double RelativeResidual{}, MassOrthogonalityError{};
    uint32_t Applications{};
};

// Returns same-discretization FP64 eigenpairs from an assembled finite-cell pencil and Accelerate Cholesky shift-invert.
// Convergence failure returns an empty result.
inline ReferenceModes AssembledOracle(const modal::FiniteCellOperator &operation, uint32_t count, double shift) {
    const auto assembled = operation.AssembleLower();
    double factor_seconds{}, solve_seconds{};
    CholeskyShiftInvert inverse{assembled.Stiffness, assembled.Mass, factor_seconds, solve_seconds};
    const uint32_t basis = std::min<uint32_t>(operation.Dofs(), count + 20);
    const auto eigensolver = modal::detail::SolveGeneralizedEigenproblem(
        inverse, assembled.Mass, assembled.Stiffness,
        {
            .Count = count,
            .SubspaceSize = basis,
            .Shift = -shift,
            .IterationTolerance = 1e-8,
            .ResidualTolerance = 1e-9,
            .MaxIterations = 1000,
            .MaxRefinementIterations = 50,
            .RandomSeed = 20260828,
        }
    );
    if (eigensolver.Eigenvalues.size() != count) return {};
    const uint32_t first_physical = std::min(6u, count);
    return {
        .Values = eigensolver.Eigenvalues,
        .Vectors = eigensolver.Eigenvectors,
        .RelativeResidual = first_physical < count ?
            eigensolver.RelativeResiduals.tail(count - first_physical).maxCoeff() :
            0,
        .MassOrthogonalityError = eigensolver.MassOrthogonalityError,
        .Applications = eigensolver.OpApplications,
    };
}
} // namespace finite_cell_benchmark
