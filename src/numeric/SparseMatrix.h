#pragma once

#include "Matrix.h"

#include <cstdint>
#include <vector>

namespace numeric {
struct Triplet {
    int Row{}, Column{};
    double Value{};
};

// Stores a compressed-column matrix. Symmetric matrices contain their lower triangle.
struct SparseMatrix {
    int Rows{}, Columns{};
    std::vector<long> ColumnStarts;
    std::vector<int> RowIndices;
    std::vector<double> Values;

    SparseMatrix() = default;
    SparseMatrix(int rows, int columns) : Rows(rows), Columns(columns), ColumnStarts(size_t(columns) + 1) {}

    int rows() const { return Rows; }
    int cols() const { return Columns; }
    size_t NonZeros() const { return Values.size(); }
    bool Empty() const { return Values.empty(); }
    const long *ColumnStartData() const { return ColumnStarts.data(); }
    const int *RowIndexData() const { return RowIndices.data(); }
    const double *ValueData() const { return Values.data(); }
    double *ValueData() { return Values.data(); }

    static SparseMatrix FromTriplets(int rows, int columns, std::vector<Triplet> triplets);
    Vector<double> Diagonal() const;
    Matrix<double> Dense() const;
    Matrix<double> DenseSymmetric() const;
};

SparseMatrix Add(const SparseMatrix &, double scale, const SparseMatrix &);
SparseMatrix Transpose(const SparseMatrix &);
SparseMatrix Multiply(const SparseMatrix &, const SparseMatrix &);
SparseMatrix ExpandSymmetric(const SparseMatrix &);
void Multiply(const SparseMatrix &, MatrixView<const double>, MatrixView<double>);
Matrix<double> Multiply(const SparseMatrix &, MatrixView<const double>);
void SymmetricMultiply(const SparseMatrix &, MatrixView<const double>, MatrixView<double>);
Matrix<double> SymmetricMultiply(const SparseMatrix &, MatrixView<const double>);
} // namespace numeric
