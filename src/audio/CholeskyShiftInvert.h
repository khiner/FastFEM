#pragma once

#include "SparseCholesky.h"

#include <Eigen/SparseCore>

#include <memory>

// Computes y = (K - sigma*M)^-1 x from the lower triangles of K and M.
// A negative sigma makes K - sigma*M positive definite when K is positive semidefinite and M is positive definite.
// The constructor stores references that accumulate factorization and solve wall-clock time.
struct CholeskyShiftInvert {
    using Scalar = double;

    CholeskyShiftInvert(
        const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m,
        double &factorize_seconds, double &solve_seconds
    );
    ~CholeskyShiftInvert();

    Eigen::Index rows() const { return K.rows(); }
    Eigen::Index cols() const { return K.cols(); }
    void set_shift(const Scalar &sigma);
    void perform_op(const Scalar *x_in, Scalar *y_out) const;
    // Solves a column-major panel of `width` right-hand sides in one factor traversal.
    void solve_panel(const Scalar *b_in, Scalar *x_out, int width) const;

    const Eigen::SparseMatrix<double> &K, &M;
    double &FactorizeSeconds, &SolveSeconds;
    std::unique_ptr<SparseCholesky> Factor;
};
