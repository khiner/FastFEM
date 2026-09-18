#pragma once

namespace numeric {
struct uvec4 {
    using value_type = unsigned;
    enum : unsigned { ComponentCount = 4 };
    unsigned x{}, y{}, z{}, w{};
    constexpr uvec4() = default;
    constexpr uvec4(unsigned v) : x(v), y(v), z(v), w(v) {}
    template<typename X, typename Y, typename Z, typename W> constexpr uvec4(X x, Y y, Z z, W w) : x(unsigned(x)), y(unsigned(y)), z(unsigned(z)), w(unsigned(w)) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(uvec4, uvec4) = default;
};

static_assert(sizeof(unsigned) == 4);
static_assert(alignof(uvec4) == 4);
static_assert(sizeof(uvec4) == 16);
} // namespace numeric
