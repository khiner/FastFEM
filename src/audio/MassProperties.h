#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <array>
#include <span>

// Stores rigid-body mass properties at the baked size and solved material density in SI units.
// The layout matches KHR_audio_modal `massProperties`, including principal moments and node-local principal axes.
struct MassProperties {
    double Mass{}; // kg
    vec3 CenterOfMass{}; // node-local units
    vec3 InertiaDiagonal{}; // principal moments, kg*m^2
    quat InertiaOrientation{}; // principal inertia axes to node-local

    bool operator==(const MassProperties &) const = default;
};

namespace modal::detail {
struct MassPropertiesAccumulator {
    double Volume{};
    dvec3 Origin{}, FirstMoment{};
    std::array<double, 6> Inertia{};
    bool Empty{true};

    void Add(dvec3 position, double volume);
    MassProperties Finish(double density, double length_to_si) const;
};

MassProperties ComputeMassProperties(std::span<const dvec3> positions, std::span<const double> volume_weights, double density, double length_to_si);
} // namespace modal::detail
