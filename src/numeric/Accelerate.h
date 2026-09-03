#pragma once

#include <cstdint>

namespace numeric {
void SymmetricCrossGram(
    const double *a, const double *b, double *result, uint32_t rows, uint32_t columns
);
bool GeneralizedSelfAdjointEigenSolve(
    double *stiffness, double *mass, double *eigenvalues, uint32_t size
);
} // namespace numeric
