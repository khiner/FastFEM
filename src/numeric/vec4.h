#pragma once

namespace numeric {

struct vec4 {
    using value_type = float;
    enum : unsigned { ComponentCount = 4 };
    float x{}, y{}, z{}, w{};
    constexpr vec4() = default;
    constexpr vec4(float v) : x(v), y(v), z(v), w(v) {}
    template<typename X, typename Y, typename Z, typename W> constexpr vec4(X x, Y y, Z z, W w) : x(float(x)), y(float(y)), z(float(z)), w(float(w)) {}
    template<typename V> constexpr vec4(const V &xyz, float w)
        requires requires { xyz.x; xyz.y; xyz.z; }
        : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}
    template<typename V> constexpr explicit vec4(const V &v)
        requires requires { v.x; v.y; v.z; v.w; }
        : x(v.x), y(v.y), z(v.z), w(v.w) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(vec4, vec4) = default;
};

static_assert(alignof(vec4) == 4);
static_assert(sizeof(vec4) == 16);

} // namespace numeric
