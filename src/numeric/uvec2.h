#pragma once

namespace numeric {
struct uvec2 {
    using value_type = unsigned;
    enum : unsigned { ComponentCount = 2 };
    unsigned x{}, y{};
    constexpr uvec2() = default;
    constexpr uvec2(unsigned v) : x(v), y(v) {}
    template<typename X, typename Y> constexpr uvec2(X x, Y y) : x(unsigned(x)), y(unsigned(y)) {}
    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return (&self.x)[i]; }
    friend constexpr bool operator==(uvec2, uvec2) = default;
};

static_assert(sizeof(unsigned) == 4);
static_assert(alignof(uvec2) == 4);
static_assert(sizeof(uvec2) == 8);
} // namespace numeric
