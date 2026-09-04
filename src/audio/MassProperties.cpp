#include "MassPropertiesAccumulator.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>

namespace {
quat QuaternionFromRotation(const Eigen::Matrix3f &rotation) {
    const auto m = [&](int column, int row) { return rotation(row, column); };
    const std::array candidates{
        m(0, 0) + m(1, 1) + m(2, 2),
        m(0, 0) - m(1, 1) - m(2, 2),
        m(1, 1) - m(0, 0) - m(2, 2),
        m(2, 2) - m(0, 0) - m(1, 1),
    };
    const auto biggest = std::ranges::max_element(candidates) - candidates.begin();
    const float value = std::sqrt(candidates[biggest] + 1.f) * .5f;
    const float scale = .25f / value;
    quat result;
    switch (biggest) {
        case 0: result = {value, (m(1, 2) - m(2, 1)) * scale, (m(2, 0) - m(0, 2)) * scale, (m(0, 1) - m(1, 0)) * scale}; break;
        case 1: result = {(m(1, 2) - m(2, 1)) * scale, value, (m(0, 1) + m(1, 0)) * scale, (m(2, 0) + m(0, 2)) * scale}; break;
        case 2: result = {(m(2, 0) - m(0, 2)) * scale, (m(0, 1) + m(1, 0)) * scale, value, (m(1, 2) + m(2, 1)) * scale}; break;
        case 3: result = {(m(0, 1) - m(1, 0)) * scale, (m(2, 0) + m(0, 2)) * scale, (m(1, 2) + m(2, 1)) * scale, value}; break;
    }
    const float norm = std::sqrt(result.w * result.w + result.x * result.x + result.y * result.y + result.z * result.z);
    return {result.w / norm, result.x / norm, result.y / norm, result.z / norm};
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
    Eigen::Matrix3d inertia;
    inertia << Inertia[0], Inertia[3], Inertia[4], Inertia[3], Inertia[1], Inertia[5], Inertia[4], Inertia[5], Inertia[2];
    const Eigen::Vector3d c{local_center.x, local_center.y, local_center.z};
    inertia -= Volume * (c.squaredNorm() * Eigen::Matrix3d::Identity() - c * c.transpose());
    inertia *= density * std::pow(length_to_si, 5);

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> decomposition{inertia};
    const auto &values = decomposition.eigenvalues();
    Eigen::Matrix3f axes = decomposition.eigenvectors().cast<float>();
    if (axes.determinant() < 0) axes.col(0) = -axes.col(0);
    return {
        density * Volume * std::pow(length_to_si, 3),
        vec3{center},
        vec3{float(values[0]), float(values[1]), float(values[2])},
        QuaternionFromRotation(axes),
    };
}

MassProperties modal::ComputeMassProperties(std::span<const dvec3> positions, std::span<const double> volume_weights, double density, double length_to_si) {
    if (positions.size() != volume_weights.size()) return {};
    MassPropertiesAccumulator accumulator;
    for (size_t i = 0; i < positions.size(); ++i) accumulator.Add(positions[i], volume_weights[i]);
    return accumulator.Finish(density, length_to_si);
}
