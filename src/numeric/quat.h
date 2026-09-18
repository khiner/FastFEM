#pragma once

namespace numeric {
struct quat {
    float x{}, y{}, z{}, w{1};
    constexpr quat() = default;
    constexpr quat(float real, float imag_x, float imag_y, float imag_z) : x(imag_x), y(imag_y), z(imag_z), w(real) {}
    template<typename V> constexpr quat(float real, const V &imaginary)
        requires requires { imaginary.x; imaginary.y; imaginary.z; }
        : x(imaginary.x), y(imaginary.y), z(imaginary.z), w(real) {}
    template<typename V> constexpr explicit quat(const V &xyzw)
        requires requires { xyzw.x; xyzw.y; xyzw.z; xyzw.w; }
        : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(quat, quat) = default;
};

static_assert(sizeof(quat) == 16);
static_assert(alignof(quat) == 4);
} // namespace numeric
