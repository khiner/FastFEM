#include "CholeskyShiftInvert.h"

#include <chrono>

namespace {
double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

CholeskyShiftInvert::CholeskyShiftInvert(
    const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m,
    double &factorize_seconds, double &solve_seconds
) : K(k), M(m), FactorizeSeconds(factorize_seconds), SolveSeconds(solve_seconds) {}

CholeskyShiftInvert::~CholeskyShiftInvert() = default;

void CholeskyShiftInvert::set_shift(const Scalar &sigma) {
    const auto start = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<double> shifted = K - sigma * M;
    Factor = std::make_unique<SparseCholesky>(shifted);
    FactorizeSeconds += SecondsSince(start);
}

void CholeskyShiftInvert::perform_op(const Scalar *x_in, Scalar *y_out) const {
    const auto start = std::chrono::steady_clock::now();
    Factor->Solve(x_in, y_out);
    SolveSeconds += SecondsSince(start);
}

void CholeskyShiftInvert::solve_panel(const Scalar *b_in, Scalar *x_out, int width) const {
    const auto start = std::chrono::steady_clock::now();
    Factor->Solve(b_in, x_out, width);
    SolveSeconds += SecondsSince(start);
}
