#pragma once

#include "vec4.h"

namespace numeric {
struct mat4 {
    vec4 Columns[4]{};

    constexpr mat4() = default;
    constexpr explicit mat4(float diagonal)
        : Columns{{diagonal, 0.f, 0.f, 0.f}, {0.f, diagonal, 0.f, 0.f}, {0.f, 0.f, diagonal, 0.f}, {0.f, 0.f, 0.f, diagonal}} {}
    constexpr mat4(vec4 c0, vec4 c1, vec4 c2, vec4 c3) : Columns{c0, c1, c2, c3} {}
    constexpr mat4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33)
        : Columns{{m00, m01, m02, m03}, {m10, m11, m12, m13}, {m20, m21, m22, m23}, {m30, m31, m32, m33}} {}

    constexpr decltype(auto) operator[](this auto &&self, unsigned i) { return self.Columns[i]; }
    friend constexpr bool operator==(mat4, mat4) = default;
};

inline constexpr mat4 I4{1.f};

static_assert(sizeof(mat4) == 64);
static_assert(alignof(mat4) == 4);
} // namespace numeric
