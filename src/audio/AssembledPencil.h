#pragma once

#include "numeric/SparseMatrix.h"

namespace modal {
// Stores the lower triangles of assembled mass and stiffness matrices.
struct AssembledPencil {
    numeric::SparseMatrix Mass;
    numeric::SparseMatrix Stiffness;
};
} // namespace modal
