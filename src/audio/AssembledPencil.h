#pragma once

#include "numeric/SparseMatrix.h"

namespace modal {
using numeric::SparseMatrix;

// Stores the lower triangles of assembled mass and stiffness matrices.
struct AssembledPencil {
    SparseMatrix Mass;
    SparseMatrix Stiffness;
};
} // namespace modal
