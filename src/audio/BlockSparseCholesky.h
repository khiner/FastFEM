#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <memory>

namespace modal {
struct Tet10Assembler;
}

struct BlockSparseCholesky {
    using Scalar = double;
    using Index = Eigen::Index;

    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit BlockSparseCholesky(const modal::Tet10Assembler &);
    ~BlockSparseCholesky();

    void SetShift(double);
    void Reassemble(const modal::Tet10Assembler &);
    void ScalePencil(double stiffness_scale, double mass_scale);
    void Solve(const double *input, double *output, int width = 1) const;

    int Size() const;

    Index rows() const { return Size(); }
    Index cols() const { return Size(); }
    void set_shift(const Scalar &sigma) { SetShift(sigma); }
    void perform_op(const Scalar *input, Scalar *output) const { Solve(input, output); }
    void solve_panel(const Scalar *input, Scalar *output, int width) const { Solve(input, output, width); }
};
