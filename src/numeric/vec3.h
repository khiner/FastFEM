#pragma once

namespace numeric {

struct vec3 {
    using value_type = float;
    enum : unsigned { ComponentCount = 3 };
    float x{}, y{}, z{};
    constexpr vec3() = default;
    constexpr vec3(float v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr vec3(X x, Y y, Z z) : x(float(x)), y(float(y)), z(float(z)) {}
    template<typename V> constexpr vec3(const V &xy, float z)
        requires requires { xy.x; xy.y; }
        : x(xy.x), y(xy.y), z(z) {}
    template<typename V> constexpr explicit vec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(vec3, vec3) = default;
};

static_assert(alignof(vec3) == 4);
static_assert(sizeof(vec3) == 12);
} // namespace numeric
