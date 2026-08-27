#pragma once

struct quat {
    float x{}, y{}, z{}, w{1.f};
    constexpr quat() = default;
    constexpr quat(float real, float imag_x, float imag_y, float imag_z) : x(imag_x), y(imag_y), z(imag_z), w(real) {}
};
constexpr bool operator==(quat a, quat b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

static_assert(sizeof(quat) == 16);
