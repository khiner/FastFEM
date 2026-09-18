#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

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
VectorView<const T> AsConst(VectorView<T> view) {
    return {view.Values, view.Count, view.Stride};
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

double Dot(VectorView<const double>, VectorView<const double>);
double Norm(VectorView<const double>);
void Scale(double, VectorView<double>);
void AddScaled(double, VectorView<const double>, VectorView<double>);
double Maximum(VectorView<const double>);
double MaximumAbsolute(VectorView<const double>);
} // namespace numeric
