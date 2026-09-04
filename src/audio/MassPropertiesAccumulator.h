#pragma once

#include "MassProperties.h"

#include <array>
#include <span>

namespace modal {
struct MassPropertiesAccumulator {
    double Volume{};
    dvec3 Origin{}, FirstMoment{};
    std::array<double, 6> Inertia{};
    bool Empty{true};

    void Add(dvec3 position, double volume);
    MassProperties Finish(double density, double length_to_si) const;
};

MassProperties ComputeMassProperties(std::span<const dvec3> positions, std::span<const double> volume_weights, double density, double length_to_si);
} // namespace modal
