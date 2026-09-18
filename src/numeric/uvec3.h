#pragma once

namespace numeric {
struct uvec3 {
    using value_type = unsigned;
    enum : unsigned { ComponentCount = 3 };
    unsigned x{}, y{}, z{};
    constexpr uvec3() = default;
    constexpr uvec3(unsigned v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr uvec3(X x, Y y, Z z) : x(unsigned(x)), y(unsigned(y)), z(unsigned(z)) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(uvec3, uvec3) = default;
};

static_assert(sizeof(unsigned) == 4);
static_assert(alignof(uvec3) == 4);
static_assert(sizeof(uvec3) == 12);
} // namespace numeric
