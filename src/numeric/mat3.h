#pragma once

#include "vec3.h"

namespace numeric {
struct mat3 {
    vec3 Columns[3]{};

    constexpr mat3() = default;
    constexpr explicit mat3(float diagonal) : Columns{{diagonal, 0.f, 0.f}, {0.f, diagonal, 0.f}, {0.f, 0.f, diagonal}} {}
    constexpr mat3(vec3 c0, vec3 c1, vec3 c2) : Columns{c0, c1, c2} {}
    constexpr mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        : Columns{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}} {}

    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return self.Columns[i]; }
    friend constexpr bool operator==(mat3, mat3) = default;
};

inline constexpr mat3 I3{1.f};

static_assert(sizeof(mat3) == 36);
static_assert(alignof(mat3) == 4);
} // namespace numeric
