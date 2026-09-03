#pragma once

#include <Eigen/SparseCore>

#include <memory>

// Reusable ordering and symbolic Cholesky analysis for matrices with identical sparsity.
struct SparseCholeskySymbolic {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit SparseCholeskySymbolic(const Eigen::SparseMatrix<double> &);
    ~SparseCholeskySymbolic();

    bool Matches(const Eigen::SparseMatrix<double> &) const;
};

// Accelerate sparse Cholesky over the lower triangle of a symmetric positive-definite matrix.
struct SparseCholesky {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit SparseCholesky(const Eigen::SparseMatrix<double> &);
    SparseCholesky(const Eigen::SparseMatrix<double> &, const SparseCholeskySymbolic &);
    ~SparseCholesky();

    void Solve(const double *input, double *output, int width = 1) const;
};
