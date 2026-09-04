#pragma once

#include "AccelerateSparseCholesky.h"

#include <Eigen/SparseCore>

#include <memory>

// Computes y = (K - sigma*M)^-1 x from the lower triangles of K and M.
// A negative sigma makes K - sigma*M positive definite when K is positive semidefinite and M is positive definite.
// The constructor stores references that accumulate factorization and solve wall-clock time.
namespace modal {
struct AccelerateShiftInvert {
    AccelerateShiftInvert(
        const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m,
        double &factorize_seconds, double &solve_seconds
    );
    ~AccelerateShiftInvert();

    void set_shift(double sigma);
    // Solves a column-major panel of `width` right-hand sides in one factor traversal.
    void solve_panel(const double *input, double *output, int width) const;

    const Eigen::SparseMatrix<double> &K, &M;
    double &FactorizeSeconds, &SolveSeconds;
    std::unique_ptr<AccelerateSparseCholesky> Factor;
};
} // namespace modal
