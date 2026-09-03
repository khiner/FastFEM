#pragma once

#include "numeric/vec3.h"

#include <cmath>
#include <cstdint>

namespace geom {
// Sign of det[a-d, b-d, c-d].
// Positive when d lies on the negative side of the plane through a, b, c (a, b, c appear counterclockwise seen from the positive side).
double Orient3D(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d);

// Returns a refined Orient3D value using the permanent from a preceding filtered evaluation.
double Orient3DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, double permanent);

// Returns det[a-d, b-d, c-d] through the cofactor evaluation order required by tetrahedralizer line-plane comparisons.
double Orient3DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d);

// Returns the 4D orientation of five points lifted to the supplied heights through the cofactor evaluation order.
double Orient4DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh);

// Returns a refined 4D orientation using the permanent from a preceding filtered evaluation.
double Orient4DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh, double permanent);

// Insphere determinant sign for the tet (a, b, c, d) and query point e.
// Requires Orient3D(a, b, c, d) > 0.
// Returns positive inside the circumsphere, negative outside, and zero when cospherical.
double InSphere(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e);

// Resolves cospherical InSphere ties by symbolic perturbation keyed on global point indices, with smaller indices dominant.
// Returns +1 or -1 for any five points spanning nonzero volume.
int InSphereSoS(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, uint32_t ia, uint32_t ib, uint32_t ic, uint32_t id, uint32_t ie);
} // namespace geom
