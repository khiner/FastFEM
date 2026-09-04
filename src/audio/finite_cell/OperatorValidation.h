#pragma once

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
numeric::Vector<double> ShiftedDiagonal(const FiniteCellOperator &, double alpha);
FiniteCellOperator::PackedCutOperators BuildPackedCutOperators(const FiniteCellOperator &, double alpha);
FiniteCellOperator WithFictitiousScale(const FiniteCellOperator &, double scale);
numeric::SparseMatrix AssembleP1ShiftedLower(const FiniteCellOperator &, double alpha);
} // namespace modal::finite_cell
