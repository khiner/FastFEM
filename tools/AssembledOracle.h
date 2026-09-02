#pragma once

#include "audio/CholeskyShiftInvert.h"
#include "audio/FiniteCell.h"

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <algorithm>

namespace finite_cell_benchmark {
struct ReferenceModes {
    Eigen::VectorXd Values;
    Eigen::MatrixXd Vectors;
    uint32_t Applications{};
};

// Same-discretization FP64 oracle: assemble the finite-cell pencil and solve it with Spectra over
// an Accelerate block Cholesky shift-invert. Empty when Spectra does not converge.
inline ReferenceModes AssembledOracle(const modal::FiniteCellOperator &operation, uint32_t count, double shift) {
    const auto assembled = operation.AssembleLower();
    double factor_seconds{}, solve_seconds{};
    CholeskyShiftInvert inverse{
        assembled.Stiffness, assembled.Mass, factor_seconds, solve_seconds,
        SparseOrdering::Metis, SparseStorage::Block3
    };
    Spectra::SparseSymMatProd<double> mass{assembled.Mass};
    const uint32_t basis = std::min<uint32_t>(operation.Dofs(), std::max(2 * count + 20, count + 40));
    Spectra::SymGEigsShiftSolver<CholeskyShiftInvert, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> eigensolver{
        inverse, mass, int(count), int(basis), -shift
    };
    eigensolver.init();
    eigensolver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestAlge);
    if (eigensolver.info() != Spectra::CompInfo::Successful) return {};
    return {
        .Values = eigensolver.eigenvalues(),
        .Vectors = eigensolver.eigenvectors(),
        .Applications = uint32_t(eigensolver.num_operations()),
    };
}
} // namespace finite_cell_benchmark
