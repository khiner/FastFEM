#pragma once

#include "MatrixView.h"

#include <vector>

namespace numeric {
template<class T>
struct Matrix {
    size_t Rows{}, Columns{};
    std::vector<T> Values;

    Matrix() = default;
    Matrix(size_t rows, size_t columns) : Rows(rows), Columns(columns), Values(rows * columns) {}

    T &operator()(size_t row, size_t column) { return Values[row + column * Rows]; }
    const T &operator()(size_t row, size_t column) const { return Values[row + column * Rows]; }
    T *data() { return Values.data(); }
    const T *data() const { return Values.data(); }
    auto begin() { return Values.begin(); }
    auto end() { return Values.end(); }
    auto begin() const { return Values.begin(); }
    auto end() const { return Values.end(); }
    size_t rows() const { return Rows; }
    size_t cols() const { return Columns; }
    size_t size() const { return Values.size(); }
    bool empty() const { return Values.empty(); }
    void Resize(size_t rows, size_t columns) {
        Rows = rows;
        Columns = columns;
        Values.resize(rows * columns);
    }
    void Clear() {
        Rows = Columns = 0;
        Values.clear();
    }
    void Fill(T value) { std::ranges::fill(Values, value); }
    void SetZero() { Fill(T{}); }
    MatrixView<T> View() { return {data(), Rows, Columns, Rows}; }
    MatrixView<const T> View() const { return {data(), Rows, Columns, Rows}; }
    VectorView<T> Column(size_t column) { return View().Column(column); }
    VectorView<const T> Column(size_t column) const { return View().Column(column); }
    MatrixView<T> FirstColumns(size_t count) { return View().FirstColumns(count); }
    MatrixView<const T> FirstColumns(size_t count) const { return View().FirstColumns(count); }
    MatrixView<T> LastColumns(size_t count) { return View().LastColumns(count); }
    MatrixView<const T> LastColumns(size_t count) const { return View().LastColumns(count); }
    MatrixView<T> ColumnsAt(size_t first, size_t count) { return View().ColumnsAt(first, count); }
    MatrixView<const T> ColumnsAt(size_t first, size_t count) const { return View().ColumnsAt(first, count); }
    MatrixView<T> Block(size_t first_row, size_t first_column, size_t rows, size_t columns) { return View().Block(first_row, first_column, rows, columns); }
    MatrixView<const T> Block(size_t first_row, size_t first_column, size_t rows, size_t columns) const { return View().Block(first_row, first_column, rows, columns); }
};

template<class T>
Matrix<T> Copy(MatrixView<const T> source) {
    Matrix<T> result(source.Rows, source.Columns);
    Copy(source, result.View());
    return result;
}

template<class T>
Matrix<T> Copy(MatrixView<T> source)
    requires(!std::is_const_v<T>)
{
    return Copy(AsConst(source));
}

template<class To, class From>
Matrix<To> Cast(MatrixView<const From> source) {
    Matrix<To> result(source.Rows, source.Columns);
    for (size_t column = 0; column < source.Columns; ++column)
        for (size_t row = 0; row < source.Rows; ++row) result(row, column) = static_cast<To>(source(row, column));
    return result;
}

template<class To, class From>
Matrix<To> Cast(MatrixView<From> source)
    requires(!std::is_const_v<From>)
{
    return Cast<To>(AsConst(source));
}

template<class T>
Matrix<T> Identity(size_t size) {
    Matrix<T> result(size, size);
    for (size_t index = 0; index < size; ++index) result(index, index) = T{1};
    return result;
}

double Norm(MatrixView<const double>);
void Scale(double, MatrixView<double>);
void AddScaled(double, MatrixView<const double>, MatrixView<double>);
Matrix<double> Multiply(MatrixView<const double>, MatrixView<const double>);
Matrix<double> TransposeMultiply(MatrixView<const double>, MatrixView<const double>);
void Multiply(MatrixView<const double>, MatrixView<const double>, MatrixView<double>, double alpha = 1, double beta = 0);
void TransposeMultiply(MatrixView<const double>, MatrixView<const double>, MatrixView<double>, double alpha = 1, double beta = 0);
void SubtractProduct(MatrixView<double>, MatrixView<const double>, MatrixView<const double>);
void ScaleColumns(MatrixView<double>, VectorView<const double>);
void ScaleRowsAndColumns(MatrixView<double>, VectorView<const double>);
void Symmetrize(MatrixView<double>);
Matrix<double> ColumnScaledDifference(MatrixView<const double>, MatrixView<const double>, VectorView<const double>);
} // namespace numeric
