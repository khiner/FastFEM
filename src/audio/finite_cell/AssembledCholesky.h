#pragma once

#include "audio/FiniteCellBlockEigensolver.h"

namespace modal::finite_cell {
FiniteCellBlockResult SolveAssembledCholesky(const FiniteCellOperator &, uint32_t count, double alpha, double tolerance = 1e-8, uint32_t max_iterations = 100);
} // namespace modal::finite_cell
