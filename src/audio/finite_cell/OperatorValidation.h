#pragma once

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
Eigen::VectorXd ShiftedDiagonal(const FiniteCellOperator &, double alpha);
FiniteCellOperator::PackedCutOperators BuildPackedCutOperators(const FiniteCellOperator &, double alpha);
FiniteCellOperator WithFictitiousScale(const FiniteCellOperator &, double scale);
Eigen::SparseMatrix<double> AssembleP1ShiftedLower(const FiniteCellOperator &, double alpha);
} // namespace modal::finite_cell
