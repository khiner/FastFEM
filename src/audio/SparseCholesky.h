#pragma once

#include <Eigen/SparseCore>

#include <memory>

enum struct SparseOrdering { Default,
                             Metis };
enum struct SparseStorage { Scalar,
                            Block3 };

// Reusable ordering and symbolic Cholesky analysis for matrices with identical sparsity.
struct SparseCholeskySymbolic {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;
    int Size{};

    explicit SparseCholeskySymbolic(
        const Eigen::SparseMatrix<double> &,
        SparseOrdering = SparseOrdering::Default,
        SparseStorage = SparseStorage::Scalar
    );
    ~SparseCholeskySymbolic();

    bool Matches(const Eigen::SparseMatrix<double> &) const;
};

// One-entry, thread-safe symbolic-analysis cache. Replacing the entry does not invalidate
// numeric factors already built from it.
struct SparseCholeskyCache {
    struct State;
    struct Acquisition {
        std::shared_ptr<const SparseCholeskySymbolic> Symbolic;
        bool Reused{};
    };

    std::unique_ptr<State> Cache;

    SparseCholeskyCache();
    ~SparseCholeskyCache();

    Acquisition Acquire(
        const Eigen::SparseMatrix<double> &,
        SparseOrdering = SparseOrdering::Default,
        SparseStorage = SparseStorage::Scalar
    );
};

// Accelerate sparse Cholesky over the lower triangle of a symmetric positive-definite matrix.
struct SparseCholesky {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;
    int Size{};

    explicit SparseCholesky(
        const Eigen::SparseMatrix<double> &,
        SparseOrdering = SparseOrdering::Default,
        SparseStorage = SparseStorage::Scalar
    );
    SparseCholesky(const Eigen::SparseMatrix<double> &, const SparseCholeskySymbolic &);
    ~SparseCholesky();

    void Solve(const double *input, double *output, int width = 1) const;
};
