#pragma once

#include "FiniteCell.h"

namespace modal {
// Records wall-clock seconds and convergence diagnostics for each solve phase.
// Other measures time outside actions, preconditioning, and Rayleigh-Ritz.
// FallbackAttempt fields describe an unconverged factor-free result before assembled Cholesky runs.
struct FiniteCellBlockProfile {
    double Total{}, Actions{}, ActionSetup{}, Initialization{}, PreconditionerSetup{}, Preconditioner{}, RayleighRitz{}, Residuals{}, Recurrence{}, Other{};
    double StagnationResidual{}, FallbackAttempt{}, FallbackAttemptResidual{};
    uint32_t StagnationIteration{}, FallbackAttemptIterations{};
    bool Stagnated{}, FallbackAttemptStagnated{};
};

struct FiniteCellBlockResult {
    Eigen::VectorXd Eigenvalues, RelativeResiduals;
    Eigen::MatrixXd Eigenvectors;
    FiniteCellBlockProfile Profile;
    uint32_t Iterations{};
};

FiniteCellBlockResult SolveFiniteCellBlock(
    const FiniteCellOperator &, uint32_t count, double alpha,
    double tolerance = 1e-8, uint32_t max_iterations = 100
);
} // namespace modal
