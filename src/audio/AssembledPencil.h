#pragma once

#include <Eigen/SparseCore>

namespace modal {
// Stores the lower triangles of assembled mass and stiffness matrices.
struct AssembledPencil {
    Eigen::SparseMatrix<double> Mass;
    Eigen::SparseMatrix<double> Stiffness;
};
} // namespace modal
