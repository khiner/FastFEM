#pragma once

#include "Matrix.h"

#include <cstdint>

namespace numeric {
void SymmetricCrossGram(
    const double *a, const double *b, double *result, uint32_t rows, uint32_t columns
);
bool GeneralizedSelfAdjointEigenSolve(
    double *stiffness, double *mass, double *eigenvalues, uint32_t size
);
bool SelfAdjointEigenSolve(double *matrix, double *eigenvalues, uint32_t size);
bool CholeskyInverse(double *matrix, uint32_t size);
bool LeastSquaresMinimumNorm(MatrixView<const double> matrix, VectorView<const double> right_hand_side, Vector<double> &solution);
bool ThinQr(Matrix<double> &);
bool SingularValues(MatrixView<const double>, Vector<double> &);
} // namespace numeric
