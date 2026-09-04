#include "MassPropertiesAccumulator.h"
#include "numeric/Accelerate.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
fastfem::Quat QuaternionFromRotation(const std::array<float, 9> &rotation) {
    const auto m = [&](int column, int row) { return rotation[row + 3 * column]; };
    const std::array candidates{
        m(0, 0) + m(1, 1) + m(2, 2),
        m(0, 0) - m(1, 1) - m(2, 2),
        m(1, 1) - m(0, 0) - m(2, 2),
        m(2, 2) - m(0, 0) - m(1, 1),
    };
    const auto biggest = std::ranges::max_element(candidates) - candidates.begin();
    const float value = std::sqrt(candidates[biggest] + 1.f) * .5f;
    const float scale = .25f / value;
    std::array<float, 4> result{};
    switch (biggest) {
        case 0: result = {(m(1, 2) - m(2, 1)) * scale, (m(2, 0) - m(0, 2)) * scale, (m(0, 1) - m(1, 0)) * scale, value}; break;
        case 1: result = {value, (m(0, 1) + m(1, 0)) * scale, (m(2, 0) + m(0, 2)) * scale, (m(1, 2) - m(2, 1)) * scale}; break;
        case 2: result = {(m(0, 1) + m(1, 0)) * scale, value, (m(1, 2) + m(2, 1)) * scale, (m(2, 0) - m(0, 2)) * scale}; break;
        case 3: result = {(m(2, 0) + m(0, 2)) * scale, (m(1, 2) + m(2, 1)) * scale, value, (m(0, 1) - m(1, 0)) * scale}; break;
    }
    const float norm = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] + result[3] * result[3]);
    for (float &component : result) component /= norm;
    return fastfem::Quat{result[3], result[0], result[1], result[2]};
}
} // namespace

void modal::MassPropertiesAccumulator::Add(dvec3 position, double volume) {
    if (Empty) {
        Origin = position;
        Empty = false;
    }
    position -= Origin;
    Volume += volume;
    FirstMoment += volume * position;
    Inertia[0] += volume * (position.y * position.y + position.z * position.z);
    Inertia[1] += volume * (position.x * position.x + position.z * position.z);
    Inertia[2] += volume * (position.x * position.x + position.y * position.y);
    Inertia[3] -= volume * position.x * position.y;
    Inertia[4] -= volume * position.x * position.z;
    Inertia[5] -= volume * position.y * position.z;
}

MassProperties modal::MassPropertiesAccumulator::Finish(double density, double length_to_si) const {
    if (Volume <= 0) return {};
    const dvec3 local_center = FirstMoment / Volume, center = Origin + local_center;
    std::array<double, 9> inertia{
        Inertia[0],
        Inertia[3],
        Inertia[4],
        Inertia[3],
        Inertia[1],
        Inertia[5],
        Inertia[4],
        Inertia[5],
        Inertia[2],
    };
    const std::array c{local_center.x, local_center.y, local_center.z};
    const double squared_norm = numeric::Dot(local_center, local_center);
    for (size_t column = 0; column < 3; ++column)
        for (size_t row = 0; row < 3; ++row)
            inertia[row + 3 * column] -= Volume * ((row == column ? squared_norm : 0) - c[row] * c[column]);
    for (double &value : inertia) value *= density * std::pow(length_to_si, 5);

    std::array<double, 3> values{};
    if (!numeric::SelfAdjointEigenSolve(inertia.data(), values.data(), 3)) return {};
    std::array<float, 9> axes;
    std::ranges::transform(inertia, axes.begin(), [](double value) { return float(value); });
    const float determinant =
        axes[0] * (axes[4] * axes[8] - axes[7] * axes[5]) -
        axes[3] * (axes[1] * axes[8] - axes[7] * axes[2]) +
        axes[6] * (axes[1] * axes[5] - axes[4] * axes[2]);
    if (determinant < 0)
        for (size_t row = 0; row < 3; ++row) axes[row] = -axes[row];
    return {
        density * Volume * std::pow(length_to_si, 3),
        fastfem::Vec3{center},
        fastfem::Vec3{float(values[0]), float(values[1]), float(values[2])},
        QuaternionFromRotation(axes),
    };
}

MassProperties modal::ComputeMassProperties(std::span<const dvec3> positions, std::span<const double> volume_weights, double density, double length_to_si) {
    if (positions.size() != volume_weights.size()) return {};
    MassPropertiesAccumulator accumulator;
    for (size_t i = 0; i < positions.size(); ++i) accumulator.Add(positions[i], volume_weights[i]);
    return accumulator.Finish(density, length_to_si);
}
