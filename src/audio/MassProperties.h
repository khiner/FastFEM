#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"

// Stores rigid-body mass properties at the baked size and solved material density in SI units.
// The layout matches KHR_audio_modal `massProperties`, including principal moments and node-local principal axes.
struct MassProperties {
    double Mass{}; // kg
    vec3 CenterOfMass{}; // node-local units
    vec3 InertiaDiagonal{}; // principal moments, kg*m^2
    quat InertiaOrientation{}; // principal inertia axes to node-local

    bool operator==(const MassProperties &) const = default;
};
