#pragma once

#include "numeric/SparseMatrix.h"

#include <memory>

// Accelerate sparse Cholesky over the lower triangle of a symmetric positive-definite matrix.
namespace modal {
struct AccelerateSparseCholesky {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit AccelerateSparseCholesky(const numeric::SparseMatrix &);
    ~AccelerateSparseCholesky();

    void Solve(const double *input, double *output, int width = 1) const;
};
} // namespace modal
