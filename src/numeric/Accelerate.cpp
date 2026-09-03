#define ACCELERATE_NEW_LAPACK

#include "numeric/Accelerate.h"

#include <Accelerate/Accelerate.h>

#include <limits>
#include <vector>

void numeric::SymmetricCrossGram(
    const double *a, const double *b, double *result, uint32_t rows, uint32_t columns
) {
    cblas_dsyr2k(CblasColMajor, CblasLower, CblasTrans, columns, rows, 0.5, a, rows, b, rows, 0, result, columns);
    for (uint32_t column = 0; column < columns; ++column)
        for (uint32_t row = 0; row < column; ++row) result[row + column * columns] = result[column + row * columns];
}

bool numeric::GeneralizedSelfAdjointEigenSolve(
    double *stiffness, double *mass, double *eigenvalues, uint32_t size
) {
    if (!size || size > uint32_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    const __LAPACK_int n = __LAPACK_int(size), problem_type = 1;
    constexpr char Eigenvectors{'V'}, Lower{'L'};
    __LAPACK_int work_size{-1}, integer_work_size{-1}, info{};
    double work_query{};
    __LAPACK_int integer_work_query{};
    dsygvd_(
        &problem_type, &Eigenvectors, &Lower, &n, stiffness, &n, mass, &n, eigenvalues,
        &work_query, &work_size, &integer_work_query, &integer_work_size, &info
    );
    if (info || work_query <= 0 || integer_work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    integer_work_size = integer_work_query;
    std::vector<double> work(static_cast<size_t>(work_size));
    std::vector<__LAPACK_int> integer_work(static_cast<size_t>(integer_work_size));
    dsygvd_(
        &problem_type, &Eigenvectors, &Lower, &n, stiffness, &n, mass, &n, eigenvalues,
        work.data(), &work_size, integer_work.data(), &integer_work_size, &info
    );
    return info == 0;
}
