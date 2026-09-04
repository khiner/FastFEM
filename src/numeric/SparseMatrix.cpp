#include "SparseMatrix.h"

#include "audio/SparseExecution.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

numeric::SparseMatrix numeric::SparseMatrix::FromTriplets(int rows, int columns, std::vector<Triplet> triplets) {
    if (rows < 0 || columns < 0) throw std::invalid_argument("Sparse matrix dimensions must be nonnegative.");
    std::vector<size_t> offsets(size_t(columns) + 1);
    for (const Triplet &entry : triplets) {
        if (entry.Row < 0 || entry.Row >= rows || entry.Column < 0 || entry.Column >= columns)
            throw std::out_of_range("Sparse triplet is outside the matrix.");
        ++offsets[size_t(entry.Column) + 1];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<size_t> next = offsets;
    std::vector<Triplet> grouped(triplets.size());
    for (const Triplet &entry : triplets) grouped[next[size_t(entry.Column)]++] = entry;
    triplets = {};

    SparseMatrix result(rows, columns);
    result.RowIndices.reserve(grouped.size());
    result.Values.reserve(grouped.size());
    for (int column = 0; column < columns; ++column) {
        auto begin = grouped.begin() + ptrdiff_t(offsets[size_t(column)]);
        const auto end = grouped.begin() + ptrdiff_t(offsets[size_t(column) + 1]);
        std::sort(begin, end, [](const Triplet &a, const Triplet &b) { return a.Row < b.Row; });
        for (auto source = begin; source != end;) {
            const int row = source->Row;
            double value{};
            do value += (source++)->Value;
            while (source != end && source->Row == row);
            if (value != 0) {
                result.RowIndices.push_back(row);
                result.Values.push_back(value);
            }
        }
        result.ColumnStarts[size_t(column) + 1] = long(result.Values.size());
    }
    return result;
}

numeric::Vector<double> numeric::SparseMatrix::Diagonal() const {
    Vector<double> result(size_t(std::min(Rows, Columns)));
    for (int column = 0; column < Columns && column < Rows; ++column)
        for (long entry = ColumnStarts[column]; entry < ColumnStarts[column + 1]; ++entry)
            if (RowIndices[entry] == column) {
                result[size_t(column)] = Values[entry];
                break;
            }
    return result;
}

numeric::Matrix<double> numeric::SparseMatrix::DenseSymmetric() const {
    if (Rows != Columns) throw std::invalid_argument("A symmetric dense conversion requires a square matrix.");
    Matrix<double> result{size_t(Rows), size_t(Columns)};
    for (int column = 0; column < Columns; ++column)
        for (long entry = ColumnStarts[column]; entry < ColumnStarts[column + 1]; ++entry) {
            const int row = RowIndices[entry];
            result(size_t(row), size_t(column)) += Values[entry];
            if (row != column) result(size_t(column), size_t(row)) += Values[entry];
        }
    return result;
}

numeric::Matrix<double> numeric::SparseMatrix::Dense() const {
    Matrix<double> result{size_t(Rows), size_t(Columns)};
    for (int column = 0; column < Columns; ++column)
        for (long entry = ColumnStarts[column]; entry < ColumnStarts[column + 1]; ++entry)
            result(size_t(RowIndices[entry]), size_t(column)) += Values[entry];
    return result;
}

numeric::SparseMatrix numeric::Add(const SparseMatrix &a, double scale, const SparseMatrix &b) {
    if (a.Rows != b.Rows || a.Columns != b.Columns) throw std::invalid_argument("Sparse matrix dimensions differ.");
    SparseMatrix result(a.Rows, a.Columns);
    result.RowIndices.reserve(a.NonZeros() + b.NonZeros());
    result.Values.reserve(a.NonZeros() + b.NonZeros());
    for (int column = 0; column < a.Columns; ++column) {
        long a_entry = a.ColumnStarts[column], b_entry = b.ColumnStarts[column];
        const long a_end = a.ColumnStarts[column + 1], b_end = b.ColumnStarts[column + 1];
        while (a_entry < a_end || b_entry < b_end) {
            const int a_row = a_entry < a_end ? a.RowIndices[a_entry] : a.Rows;
            const int b_row = b_entry < b_end ? b.RowIndices[b_entry] : b.Rows;
            const int row = std::min(a_row, b_row);
            double value{};
            if (a_row == row) value += a.Values[a_entry++];
            if (b_row == row) value += scale * b.Values[b_entry++];
            if (value != 0) {
                result.RowIndices.push_back(row);
                result.Values.push_back(value);
            }
        }
        result.ColumnStarts[size_t(column) + 1] = long(result.Values.size());
    }
    return result;
}

numeric::SparseMatrix numeric::Transpose(const SparseMatrix &matrix) {
    std::vector<Triplet> entries;
    entries.reserve(matrix.NonZeros());
    for (int column = 0; column < matrix.Columns; ++column)
        for (long entry = matrix.ColumnStarts[column]; entry < matrix.ColumnStarts[column + 1]; ++entry)
            entries.push_back({column, matrix.RowIndices[entry], matrix.Values[entry]});
    return SparseMatrix::FromTriplets(matrix.Columns, matrix.Rows, std::move(entries));
}

numeric::SparseMatrix numeric::ExpandSymmetric(const SparseMatrix &matrix) {
    if (matrix.Rows != matrix.Columns) throw std::invalid_argument("Symmetric expansion requires a square matrix.");
    std::vector<Triplet> entries;
    entries.reserve(2 * matrix.NonZeros());
    for (int column = 0; column < matrix.Columns; ++column)
        for (long entry = matrix.ColumnStarts[column]; entry < matrix.ColumnStarts[column + 1]; ++entry) {
            const int row = matrix.RowIndices[entry];
            entries.push_back({row, column, matrix.Values[entry]});
            if (row != column) entries.push_back({column, row, matrix.Values[entry]});
        }
    return SparseMatrix::FromTriplets(matrix.Rows, matrix.Columns, std::move(entries));
}

numeric::SparseMatrix numeric::Multiply(const SparseMatrix &a, const SparseMatrix &b) {
    if (a.Columns != b.Rows) throw std::invalid_argument("Sparse matrix-product dimensions differ.");
    std::vector<Triplet> entries;
    std::vector<double> accumulator(size_t(a.Rows));
    std::vector<uint32_t> marks(size_t(a.Rows));
    std::vector<int> touched;
    for (int column = 0; column < b.Columns; ++column) {
        touched.clear();
        const uint32_t mark = uint32_t(column) + 1;
        for (long b_entry = b.ColumnStarts[column]; b_entry < b.ColumnStarts[column + 1]; ++b_entry) {
            const int inner = b.RowIndices[b_entry];
            for (long a_entry = a.ColumnStarts[inner]; a_entry < a.ColumnStarts[inner + 1]; ++a_entry) {
                const int row = a.RowIndices[a_entry];
                if (marks[size_t(row)] != mark) {
                    marks[size_t(row)] = mark;
                    touched.push_back(row);
                }
                accumulator[size_t(row)] += a.Values[a_entry] * b.Values[b_entry];
            }
        }
        std::ranges::sort(touched);
        for (int row : touched) {
            entries.push_back({row, column, accumulator[size_t(row)]});
            accumulator[size_t(row)] = 0;
        }
    }
    return SparseMatrix::FromTriplets(a.Rows, b.Columns, std::move(entries));
}

void numeric::Multiply(const SparseMatrix &matrix, MatrixView<const double> input, MatrixView<double> output) {
    if (size_t(matrix.Columns) != input.Rows || size_t(matrix.Rows) != output.Rows || output.Columns != input.Columns)
        throw std::invalid_argument("Sparse matrix-product dimensions differ.");
    ConfigureAccelerateSparseExecution();
    const SparseMatrix_Double sparse{
        .structure = {
            .rowCount = matrix.Rows,
            .columnCount = matrix.Columns,
            .columnStarts = const_cast<long *>(matrix.ColumnStarts.data()),
            .rowIndices = const_cast<int *>(matrix.RowIndices.data()),
            .attributes = {},
            .blockSize = 1,
        },
        .data = const_cast<double *>(matrix.Values.data()),
    };
    const DenseMatrix_Double source{int(input.Rows), int(input.Columns), int(input.LeadingDimension), SparseAttributes_t{}, const_cast<double *>(input.Values)};
    const DenseMatrix_Double destination{int(output.Rows), int(output.Columns), int(output.LeadingDimension), SparseAttributes_t{}, output.Values};
    SparseMultiply(sparse, source, destination);
}

numeric::Matrix<double> numeric::Multiply(const SparseMatrix &matrix, MatrixView<const double> input) {
    Matrix<double> result(size_t(matrix.Rows), input.Columns);
    Multiply(matrix, input, result.View());
    return result;
}

void numeric::SymmetricMultiply(const SparseMatrix &matrix, MatrixView<const double> input, MatrixView<double> output) {
    if (matrix.Rows != matrix.Columns || size_t(matrix.Columns) != input.Rows || output.Rows != input.Rows || output.Columns != input.Columns)
        throw std::invalid_argument("Symmetric sparse matrix-product dimensions differ.");
    for (size_t panel = 0; panel < input.Columns; ++panel) {
        const double *source = input.Values + panel * input.LeadingDimension;
        double *destination = output.Values + panel * output.LeadingDimension;
        std::fill_n(destination, output.Rows, 0.0);
        for (int column = 0; column < matrix.Columns; ++column) {
            const double source_column = source[column];
            long entry = matrix.ColumnStarts[column];
            const long end = matrix.ColumnStarts[column + 1];
            double column_sum{};
            if (entry < end && matrix.RowIndices[entry] == column) column_sum = matrix.Values[entry++] * source_column;
            for (; entry < end; ++entry) {
                const int row = matrix.RowIndices[entry];
                const double value = matrix.Values[entry];
                destination[row] += value * source_column;
                column_sum += value * source[row];
            }
            destination[column] += column_sum;
        }
    }
}

numeric::Matrix<double> numeric::SymmetricMultiply(const SparseMatrix &matrix, MatrixView<const double> input) {
    Matrix<double> result(size_t(matrix.Rows), input.Columns);
    SymmetricMultiply(matrix, input, result.View());
    return result;
}
