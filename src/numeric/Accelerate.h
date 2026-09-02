#pragma once

#include <cstdint>

namespace numeric {
void SymmetricCrossGram(
    const double *a, const double *b, double *result, uint32_t rows, uint32_t columns
);
}
