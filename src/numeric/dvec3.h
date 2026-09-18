#pragma once

namespace numeric {
struct dvec3 {
    using value_type = double;
    enum : unsigned { ComponentCount = 3 };
    double x{}, y{}, z{};
    constexpr dvec3() = default;
    constexpr dvec3(double v) : x(v), y(v), z(v) {}
    constexpr dvec3(double x, double y, double z) : x(x), y(y), z(z) {}
    template<typename V> constexpr explicit dvec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(dvec3, dvec3) = default;
};

static_assert(alignof(dvec3) == 8);
static_assert(sizeof(dvec3) == 24);
} // namespace numeric
