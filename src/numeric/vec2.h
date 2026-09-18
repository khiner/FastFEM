#pragma once

namespace numeric {

struct vec2 {
    using value_type = float;
    enum : unsigned { ComponentCount = 2 };
    float x{}, y{};
    constexpr vec2() = default;
    constexpr vec2(float v) : x(v), y(v) {}
    template<typename X, typename Y> constexpr vec2(X x, Y y) : x(float(x)), y(float(y)) {}
    template<typename V> constexpr vec2(const V &v)
        requires requires { v.x; v.y; }
        : x(v.x), y(v.y) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(vec2, vec2) = default;
};

static_assert(alignof(vec2) == 4);
static_assert(sizeof(vec2) == 8);
} // namespace numeric
