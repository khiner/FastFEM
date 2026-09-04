#pragma once

#include <Eigen/SparseCore>

#include <memory>

// Accelerate sparse Cholesky over the lower triangle of a symmetric positive-definite matrix.
struct SparseCholesky {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit SparseCholesky(const Eigen::SparseMatrix<double> &);
    ~SparseCholesky();

    void Solve(const double *input, double *output, int width = 1) const;
};
