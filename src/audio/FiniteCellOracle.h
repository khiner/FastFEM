#pragma once

#include "FiniteCellBlockEigensolver.h"

namespace modal::oracle {
FiniteCellOperator BuildOctree(const ImplicitDomain &, const AcousticMaterialProperties &, FiniteCellConfig = {});
FiniteCellBlockResult SolveCholesky(const FiniteCellOperator &, uint32_t count, double alpha, double tolerance = 1e-8, uint32_t max_iterations = 100);
} // namespace modal::oracle
