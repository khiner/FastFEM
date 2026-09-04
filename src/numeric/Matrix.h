#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace numeric {
template<class T>
struct VectorView {
    T *Values{};
    size_t Count{}, Stride{1};

    T &operator[](size_t index) const { return Values[index * Stride]; }
    T *data() const { return Values; }
    size_t size() const { return Count; }
    VectorView First(size_t count) const { return {Values, count, Stride}; }
    VectorView Last(size_t count) const { return {Values + (Count - count) * Stride, count, Stride}; }
    VectorView Subvector(size_t first, size_t count) const { return {Values + first * Stride, count, Stride}; }
    operator VectorView<const T>() const
        requires(!std::is_const_v<T>)
    { return {Values, Count, Stride}; }
};

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
VectorView<const T> AsConst(VectorView<T> view) {
    return {view.Values, view.Count, view.Stride};
}

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
struct Vector {
    std::vector<T> Values;

    Vector() = default;
    explicit Vector(size_t size) : Values(size) {}
    Vector(size_t size, T value) : Values(size, value) {}

    T &operator[](size_t index) { return Values[index]; }
    const T &operator[](size_t index) const { return Values[index]; }
    T *data() { return Values.data(); }
    const T *data() const { return Values.data(); }
    auto begin() { return Values.begin(); }
    auto end() { return Values.end(); }
    auto begin() const { return Values.begin(); }
    auto end() const { return Values.end(); }
    size_t size() const { return Values.size(); }
    bool empty() const { return Values.empty(); }
    void Resize(size_t size) { Values.resize(size); }
    void Clear() { Values.clear(); }
    void Fill(T value) { std::ranges::fill(Values, value); }
    void SetZero() { Fill(T{}); }
    VectorView<T> View() { return {data(), size(), 1}; }
    VectorView<const T> View() const { return {data(), size(), 1}; }
    VectorView<T> First(size_t count) { return View().First(count); }
    VectorView<const T> First(size_t count) const { return View().First(count); }
    VectorView<T> Last(size_t count) { return View().Last(count); }
    VectorView<const T> Last(size_t count) const { return View().Last(count); }
    VectorView<T> Subvector(size_t first, size_t count) { return View().Subvector(first, count); }
    VectorView<const T> Subvector(size_t first, size_t count) const { return View().Subvector(first, count); }
};

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
void Copy(VectorView<const T> source, VectorView<T> destination) {
    assert(source.Count == destination.Count);
    for (size_t index = 0; index < source.Count; ++index) destination[index] = source[index];
}

template<class T>
void Copy(VectorView<T> source, VectorView<T> destination)
    requires(!std::is_const_v<T>)
{
    Copy(AsConst(source), destination);
}

template<class T>
Vector<T> Copy(VectorView<const T> source) {
    Vector<T> result(source.Count);
    Copy(source, result.View());
    return result;
}

template<class T>
Vector<T> Copy(VectorView<T> source)
    requires(!std::is_const_v<T>)
{
    return Copy(AsConst(source));
}

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

template<class T>
bool AllFinite(VectorView<const T> vector) {
    for (size_t index = 0; index < vector.Count; ++index)
        if (!std::isfinite(vector[index])) return false;
    return true;
}

template<class T>
bool AllFinite(VectorView<T> vector)
    requires(!std::is_const_v<T>)
{
    return AllFinite(AsConst(vector));
}

template<class T>
Matrix<T> Identity(size_t size) {
    Matrix<T> result(size, size);
    for (size_t index = 0; index < size; ++index) result(index, index) = T{1};
    return result;
}

double Dot(VectorView<const double>, VectorView<const double>);
double Norm(VectorView<const double>);
double Norm(MatrixView<const double>);
void Scale(double, VectorView<double>);
void Scale(double, MatrixView<double>);
void AddScaled(double, VectorView<const double>, VectorView<double>);
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
double Maximum(VectorView<const double>);
double MaximumAbsolute(VectorView<const double>);
} // namespace numeric
