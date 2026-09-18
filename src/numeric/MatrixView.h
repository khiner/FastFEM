#pragma once

#include "VectorView.h"

#include <algorithm>

namespace numeric {
template<class T>
struct MatrixView {
    T *Values{};
    size_t Rows{}, Columns{}, LeadingDimension{};

    T &operator()(size_t row, size_t column) const { return Values[row + column * LeadingDimension]; }
    T *data() const { return Values; }
    size_t rows() const { return Rows; }
    size_t cols() const { return Columns; }
    size_t size() const { return Rows * Columns; }
    VectorView<T> Column(size_t column) const { return {Values + column * LeadingDimension, Rows, 1}; }
    MatrixView FirstColumns(size_t count) const { return {Values, Rows, count, LeadingDimension}; }
    MatrixView LastColumns(size_t count) const { return {Values + (Columns - count) * LeadingDimension, Rows, count, LeadingDimension}; }
    MatrixView ColumnsAt(size_t first, size_t count) const { return {Values + first * LeadingDimension, Rows, count, LeadingDimension}; }
    MatrixView Block(size_t first_row, size_t first_column, size_t rows, size_t columns) const {
        return {Values + first_row + first_column * LeadingDimension, rows, columns, LeadingDimension};
    }
    operator MatrixView<const T>() const
        requires(!std::is_const_v<T>)
    { return {Values, Rows, Columns, LeadingDimension}; }
};

template<class T>
MatrixView<const T> AsConst(MatrixView<T> view) {
    return {view.Values, view.Rows, view.Columns, view.LeadingDimension};
}

template<class T>
void Copy(MatrixView<const T> source, MatrixView<T> destination) {
    assert(source.Rows == destination.Rows && source.Columns == destination.Columns);
    for (size_t column = 0; column < source.Columns; ++column)
        std::copy_n(source.Values + column * source.LeadingDimension, source.Rows, destination.Values + column * destination.LeadingDimension);
}

template<class T>
void Copy(MatrixView<T> source, MatrixView<T> destination)
    requires(!std::is_const_v<T>)
{
    Copy(AsConst(source), destination);
}

template<class T>
bool AllFinite(MatrixView<const T> matrix) {
    for (size_t column = 0; column < matrix.Columns; ++column)
        for (size_t row = 0; row < matrix.Rows; ++row)
            if (!std::isfinite(matrix(row, column))) return false;
    return true;
}

template<class T>
bool AllFinite(MatrixView<T> matrix)
    requires(!std::is_const_v<T>)
{
    return AllFinite(AsConst(matrix));
}
} // namespace numeric
