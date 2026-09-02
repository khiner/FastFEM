#define ACCELERATE_NEW_LAPACK

#include "numeric/Accelerate.h"

#include <Accelerate/Accelerate.h>

void numeric::SymmetricCrossGram(
    const double *a, const double *b, double *result, uint32_t rows, uint32_t columns
) {
    cblas_dsyr2k(CblasColMajor, CblasLower, CblasTrans, columns, rows, 0.5, a, rows, b, rows, 0, result, columns);
    for (uint32_t column = 0; column < columns; ++column)
        for (uint32_t row = 0; row < column; ++row) result[row + column * columns] = result[column + row * columns];
}
