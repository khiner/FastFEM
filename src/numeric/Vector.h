#pragma once

#include "VectorView.h"

#include <algorithm>
#include <vector>

namespace numeric {
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
} // namespace numeric
