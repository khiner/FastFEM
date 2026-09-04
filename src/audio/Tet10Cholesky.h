#pragma once

#include <memory>

namespace modal {
struct Tet10Assembler;

struct Tet10Cholesky {
    struct Factorization;

    std::unique_ptr<Factorization> Factor;

    explicit Tet10Cholesky(const modal::Tet10Assembler &);
    ~Tet10Cholesky();

    void SetShift(double);
    void Reassemble(const modal::Tet10Assembler &);
    void ScalePencil(double stiffness_scale, double mass_scale);
    void Solve(const double *input, double *output, int width = 1) const;

    void set_shift(double sigma) { SetShift(sigma); }
    void solve_panel(const double *input, double *output, int width) const { Solve(input, output, width); }
};
} // namespace modal
