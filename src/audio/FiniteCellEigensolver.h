#pragma once

#include "FiniteCell.h"

namespace modal {
// Records wall-clock seconds and convergence diagnostics for each solve phase.
// Other measures time outside actions, preconditioning, and Rayleigh-Ritz.
// A failed factor-free solve is recorded when assembled Cholesky runs.
struct FiniteCellSolveProfile {
    double Total{}, Actions{}, ActionSetup{}, Initialization{}, PreconditionerSetup{}, Preconditioner{}, RayleighRitz{}, Residuals{}, Recurrence{}, Other{};
    double StagnationResidual{}, FailedFactorFreeSeconds{}, FailedFactorFreeResidual{};
    uint32_t StagnationIteration{}, FailedFactorFreeIterations{};
    bool Stagnated{}, FailedFactorFreeStagnated{};
};

struct FiniteCellEigenpairs {
    numeric::Vector<double> Eigenvalues, RelativeResiduals;
    numeric::Matrix<double> Eigenvectors;
    FiniteCellSolveProfile Profile;
    uint32_t Iterations{};
};

FiniteCellEigenpairs SolveFiniteCellEigenpairs(
    const FiniteCellOperator &, uint32_t count, double alpha,
    double tolerance = 1e-8, uint32_t max_iterations = 100
);
} // namespace modal
