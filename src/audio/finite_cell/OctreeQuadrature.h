#pragma once

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
FiniteCellOperator BuildOctreeOperator(const ImplicitDomain &, const AcousticMaterialProperties &, FiniteCellConfig = {});
} // namespace modal::finite_cell
