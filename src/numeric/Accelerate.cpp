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

bool numeric::SelfAdjointEigenSolve(double *matrix, double *eigenvalues, uint32_t size) {
    if (!size || size > uint32_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    const __LAPACK_int n = __LAPACK_int(size);
    constexpr char Eigenvectors{'V'}, Lower{'L'};
    __LAPACK_int work_size{-1}, info{};
    double work_query{};
    dsyev_(&Eigenvectors, &Lower, &n, matrix, &n, eigenvalues, &work_query, &work_size, &info);
    if (info || work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    std::vector<double> work(static_cast<size_t>(work_size));
    dsyev_(&Eigenvectors, &Lower, &n, matrix, &n, eigenvalues, work.data(), &work_size, &info);
    return info == 0;
}

bool numeric::CholeskyInverse(double *matrix, uint32_t size) {
    if (!size || size > uint32_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    const __LAPACK_int n = __LAPACK_int(size);
    constexpr char Lower{'L'};
    __LAPACK_int info{};
    dpotrf_(&Lower, &n, matrix, &n, &info);
    if (info) return false;
    dpotri_(&Lower, &n, matrix, &n, &info);
    if (info) return false;
    for (uint32_t column = 0; column < size; ++column)
        for (uint32_t row = 0; row < column; ++row) matrix[row + size * column] = matrix[column + size * row];
    return true;
}

bool numeric::LeastSquaresMinimumNorm(
    MatrixView<const double> matrix, VectorView<const double> right_hand_side, Vector<double> &solution
) {
    if (!matrix.Rows || !matrix.Columns || matrix.Rows != right_hand_side.Count ||
        matrix.Rows > size_t(std::numeric_limits<__LAPACK_int>::max()) ||
        matrix.Columns > size_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    Matrix<double> copy = Copy(matrix);
    const __LAPACK_int rows = __LAPACK_int(matrix.Rows), columns = __LAPACK_int(matrix.Columns), right_hand_sides = 1;
    const __LAPACK_int leading_rhs = std::max(rows, columns);
    std::vector<double> rhs((size_t(leading_rhs)));
    std::copy_n(right_hand_side.Values, right_hand_side.Count, rhs.begin());
    std::vector<__LAPACK_int> pivots(matrix.Columns);
    constexpr double RankTolerance{-1};
    __LAPACK_int rank{}, work_size{-1}, info{};
    double work_query{};
    dgelsy_(&rows, &columns, &right_hand_sides, copy.data(), &rows, rhs.data(), &leading_rhs, pivots.data(), &RankTolerance, &rank, &work_query, &work_size, &info);
    if (info || work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    std::vector<double> work(static_cast<size_t>(work_size));
    dgelsy_(&rows, &columns, &right_hand_sides, copy.data(), &rows, rhs.data(), &leading_rhs, pivots.data(), &RankTolerance, &rank, work.data(), &work_size, &info);
    if (info) return false;
    solution.Resize(matrix.Columns);
    std::copy_n(rhs.begin(), matrix.Columns, solution.Values.begin());
    return true;
}

bool numeric::ThinQr(Matrix<double> &matrix) {
    if (!matrix.Rows || !matrix.Columns || matrix.Rows < matrix.Columns ||
        matrix.Rows > size_t(std::numeric_limits<__LAPACK_int>::max()) ||
        matrix.Columns > size_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    const __LAPACK_int rows = __LAPACK_int(matrix.Rows), columns = __LAPACK_int(matrix.Columns);
    std::vector<double> reflectors(matrix.Columns);
    __LAPACK_int work_size{-1}, info{};
    double work_query{};
    dgeqrf_(&rows, &columns, matrix.data(), &rows, reflectors.data(), &work_query, &work_size, &info);
    if (info || work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    std::vector<double> work(static_cast<size_t>(work_size));
    dgeqrf_(&rows, &columns, matrix.data(), &rows, reflectors.data(), work.data(), &work_size, &info);
    if (info) return false;
    work_size = -1;
    dorgqr_(&rows, &columns, &columns, matrix.data(), &rows, reflectors.data(), &work_query, &work_size, &info);
    if (info || work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    work.resize(static_cast<size_t>(work_size));
    dorgqr_(&rows, &columns, &columns, matrix.data(), &rows, reflectors.data(), work.data(), &work_size, &info);
    return info == 0;
}

bool numeric::SingularValues(MatrixView<const double> matrix, Vector<double> &values) {
    if (!matrix.Rows || !matrix.Columns || matrix.Rows > size_t(std::numeric_limits<__LAPACK_int>::max()) ||
        matrix.Columns > size_t(std::numeric_limits<__LAPACK_int>::max())) return false;
    Matrix<double> copy = Copy(matrix);
    const __LAPACK_int rows = __LAPACK_int(matrix.Rows), columns = __LAPACK_int(matrix.Columns);
    constexpr char None{'N'};
    const __LAPACK_int unused_dimension{1};
    double unused{};
    values.Resize(std::min(matrix.Rows, matrix.Columns));
    __LAPACK_int work_size{-1}, info{};
    double work_query{};
    dgesvd_(&None, &None, &rows, &columns, copy.data(), &rows, values.data(), &unused, &unused_dimension, &unused, &unused_dimension, &work_query, &work_size, &info);
    if (info || work_query <= 0) return false;
    work_size = __LAPACK_int(work_query);
    std::vector<double> work(static_cast<size_t>(work_size));
    dgesvd_(&None, &None, &rows, &columns, copy.data(), &rows, values.data(), &unused, &unused_dimension, &unused, &unused_dimension, work.data(), &work_size, &info);
    return info == 0;
}
