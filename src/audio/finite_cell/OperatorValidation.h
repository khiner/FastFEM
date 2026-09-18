#pragma once

#include "numeric/Vector.h"

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
using numeric::Vector;

Vector<double> ShiftedDiagonal(const FiniteCellOperator &, double alpha);
FiniteCellOperator::PackedCutOperators BuildPackedCutOperators(const FiniteCellOperator &, double alpha);
FiniteCellOperator WithFictitiousScale(const FiniteCellOperator &, double scale);
SparseMatrix AssembleP1ShiftedLower(const FiniteCellOperator &, double alpha);
} // namespace modal::finite_cell
