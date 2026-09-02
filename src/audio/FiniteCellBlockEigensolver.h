#pragma once

#include "FiniteCell.h"

namespace modal {
// Wall-clock seconds per solve phase. `Other` is the remainder outside actions, preconditioning,
// and Rayleigh-Ritz. The FallbackAttempt fields describe a factor-free attempt that failed
// certification and handed the solve to the assembled Cholesky route.
struct FiniteCellBlockProfile {
    double Total{}, Actions{}, ActionSetup{}, Initialization{}, PreconditionerSetup{}, Preconditioner{},
        RayleighRitz{}, Residuals{}, Recurrence{}, Certification{}, Other{};
    double StagnationResidual{}, FallbackAttempt{}, FallbackAttemptResidual{}, FallbackAttemptOrthogonality{};
    uint32_t StagnationIteration{}, FallbackAttemptIterations{};
    bool Stagnated{}, FallbackAttemptStagnated{};
};

struct FiniteCellBlockResult {
    Eigen::VectorXd Eigenvalues, RelativeResiduals;
    Eigen::MatrixXd Eigenvectors;
    FiniteCellCertification Certification;
    FiniteCellBlockProfile Profile;
    uint32_t Iterations{};
};

FiniteCellBlockResult SolveFiniteCellBlock(
    const FiniteCellOperator &, uint32_t count, double alpha,
    double tolerance = 1e-8, uint32_t max_iterations = 100
);
FiniteCellBlockResult SolveFiniteCellBlockCholesky(
    const FiniteCellOperator &, uint32_t count, double alpha,
    double tolerance = 1e-8, uint32_t max_iterations = 100
);
} // namespace modal
