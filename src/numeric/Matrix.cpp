#include "Matrix.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <limits>
#include <stdexcept>

namespace {
__LAPACK_int LapackSize(size_t value) {
    if (value > size_t(std::numeric_limits<__LAPACK_int>::max())) throw std::length_error("Matrix dimension exceeds the Accelerate LP64 interface.");
    return __LAPACK_int(value);
}
} // namespace

double numeric::Dot(VectorView<const double> a, VectorView<const double> b) {
    if (a.Count != b.Count) throw std::invalid_argument("Dot-product vector lengths differ.");
    return cblas_ddot(LapackSize(a.Count), a.Values, LapackSize(a.Stride), b.Values, LapackSize(b.Stride));
}

double numeric::Norm(VectorView<const double> vector) {
    return cblas_dnrm2(LapackSize(vector.Count), vector.Values, LapackSize(vector.Stride));
}

double numeric::Norm(MatrixView<const double> matrix) {
    double squared{};
    for (size_t column = 0; column < matrix.Columns; ++column) {
        const double norm = cblas_dnrm2(LapackSize(matrix.Rows), matrix.Values + column * matrix.LeadingDimension, 1);
        squared += norm * norm;
    }
    return std::sqrt(squared);
}

void numeric::Scale(double scale, VectorView<double> vector) {
    cblas_dscal(LapackSize(vector.Count), scale, vector.Values, LapackSize(vector.Stride));
}

void numeric::Scale(double scale, MatrixView<double> matrix) {
    for (size_t column = 0; column < matrix.Columns; ++column)
        cblas_dscal(LapackSize(matrix.Rows), scale, matrix.Values + column * matrix.LeadingDimension, 1);
}

void numeric::AddScaled(double scale, VectorView<const double> source, VectorView<double> destination) {
    if (source.Count != destination.Count) throw std::invalid_argument("AXPY vector lengths differ.");
    cblas_daxpy(LapackSize(source.Count), scale, source.Values, LapackSize(source.Stride), destination.Values, LapackSize(destination.Stride));
}

void numeric::AddScaled(double scale, MatrixView<const double> source, MatrixView<double> destination) {
    if (source.Rows != destination.Rows || source.Columns != destination.Columns) throw std::invalid_argument("AXPY matrix dimensions differ.");
    for (size_t column = 0; column < source.Columns; ++column)
        cblas_daxpy(LapackSize(source.Rows), scale, source.Values + column * source.LeadingDimension, 1, destination.Values + column * destination.LeadingDimension, 1);
}

void numeric::Multiply(
    MatrixView<const double> a, MatrixView<const double> b, MatrixView<double> result, double alpha, double beta
) {
    if (a.Columns != b.Rows || result.Rows != a.Rows || result.Columns != b.Columns) throw std::invalid_argument("Matrix-product dimensions differ.");
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, LapackSize(a.Rows), LapackSize(b.Columns), LapackSize(a.Columns), alpha, a.Values, LapackSize(a.LeadingDimension), b.Values, LapackSize(b.LeadingDimension), beta, result.Values, LapackSize(result.LeadingDimension));
}

void numeric::TransposeMultiply(
    MatrixView<const double> a, MatrixView<const double> b, MatrixView<double> result, double alpha, double beta
) {
    if (a.Rows != b.Rows || result.Rows != a.Columns || result.Columns != b.Columns) throw std::invalid_argument("Transposed matrix-product dimensions differ.");
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, LapackSize(a.Columns), LapackSize(b.Columns), LapackSize(a.Rows), alpha, a.Values, LapackSize(a.LeadingDimension), b.Values, LapackSize(b.LeadingDimension), beta, result.Values, LapackSize(result.LeadingDimension));
}

numeric::Matrix<double> numeric::Multiply(MatrixView<const double> a, MatrixView<const double> b) {
    Matrix<double> result(a.Rows, b.Columns);
    Multiply(a, b, result.View());
    return result;
}

numeric::Matrix<double> numeric::TransposeMultiply(MatrixView<const double> a, MatrixView<const double> b) {
    Matrix<double> result(a.Columns, b.Columns);
    TransposeMultiply(a, b, result.View());
    return result;
}

void numeric::SubtractProduct(MatrixView<double> destination, MatrixView<const double> a, MatrixView<const double> b) {
    Multiply(a, b, destination, -1, 1);
}

void numeric::ScaleColumns(MatrixView<double> matrix, VectorView<const double> scales) {
    if (matrix.Columns != scales.Count) throw std::invalid_argument("Column-scale dimensions differ.");
    for (size_t column = 0; column < matrix.Columns; ++column)
        cblas_dscal(LapackSize(matrix.Rows), scales[column], matrix.Values + column * matrix.LeadingDimension, 1);
}

void numeric::ScaleRowsAndColumns(MatrixView<double> matrix, VectorView<const double> scales) {
    if (matrix.Rows != scales.Count || matrix.Columns != scales.Count) throw std::invalid_argument("Symmetric matrix-scale dimensions differ.");
    for (size_t column = 0; column < matrix.Columns; ++column)
        for (size_t row = 0; row < matrix.Rows; ++row) matrix(row, column) *= scales[row] * scales[column];
}

void numeric::Symmetrize(MatrixView<double> matrix) {
    if (matrix.Rows != matrix.Columns) throw std::invalid_argument("Symmetrization requires a square matrix.");
    for (size_t column = 0; column < matrix.Columns; ++column)
        for (size_t row = column + 1; row < matrix.Rows; ++row) {
            const double value = 0.5 * (matrix(row, column) + matrix(column, row));
            matrix(row, column) = matrix(column, row) = value;
        }
}

numeric::Matrix<double> numeric::ColumnScaledDifference(
    MatrixView<const double> left, MatrixView<const double> right, VectorView<const double> scales
) {
    if (left.Rows != right.Rows || left.Columns != right.Columns || left.Columns != scales.Count)
        throw std::invalid_argument("Column-scaled difference dimensions differ.");
    Matrix<double> result = Copy(left);
    for (size_t column = 0; column < left.Columns; ++column)
        cblas_daxpy(LapackSize(left.Rows), -scales[column], right.Values + column * right.LeadingDimension, 1, result.data() + column * result.Rows, 1);
    return result;
}

double numeric::Maximum(VectorView<const double> vector) {
    if (!vector.Count) return -std::numeric_limits<double>::infinity();
    double result = vector[0];
    for (size_t index = 1; index < vector.Count; ++index) result = std::max(result, vector[index]);
    return result;
}

double numeric::MaximumAbsolute(VectorView<const double> vector) {
    double result{};
    for (size_t index = 0; index < vector.Count; ++index) result = std::max(result, std::abs(vector[index]));
    return result;
}
