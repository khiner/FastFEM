#include "AccelerateShiftInvert.h"

#include <chrono>

namespace {
double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

modal::AccelerateShiftInvert::AccelerateShiftInvert(
    const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m,
    double &factorize_seconds, double &solve_seconds
) : K(k), M(m), FactorizeSeconds(factorize_seconds), SolveSeconds(solve_seconds) {}

modal::AccelerateShiftInvert::~AccelerateShiftInvert() = default;

void modal::AccelerateShiftInvert::set_shift(double sigma) {
    const auto start = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<double> shifted = K - sigma * M;
    Factor = std::make_unique<AccelerateSparseCholesky>(shifted);
    FactorizeSeconds += SecondsSince(start);
}

void modal::AccelerateShiftInvert::solve_panel(const double *input, double *output, int width) const {
    const auto start = std::chrono::steady_clock::now();
    Factor->Solve(input, output, width);
    SolveSeconds += SecondsSince(start);
}
