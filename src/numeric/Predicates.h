#pragma once

#include "numeric/vec3.h"

#include <cmath>
#include <cstdint>

namespace geom {
// Sign of det[a-d, b-d, c-d].
// Positive when d lies on the negative side of the plane through a, b, c (a, b, c appear counterclockwise seen from the positive side).
double Orient3D(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d);

double Orient3DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, double permanent);

// The same determinant through the cofactor pipeline the tetrahedralizer's line-plane intersection compares values from.
// It returns the expansion's leading component, which differs from the expansion Orient3DRefined ends at in the last bit on about a fifth of inputs.
double Orient3DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d);

// The 4d orientation of five points lifted to the given heights, through the same pipeline.
double Orient4DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh);

double Orient4DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh, double permanent);

double InSphere(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e);

// InSphere with symbolic perturbation of cospherical ties, keyed on global point indices (smaller index dominates).
// Returns +1 or -1 for any five points spanning nonzero volume.
int InSphereSoS(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, uint32_t ia, uint32_t ib, uint32_t ic, uint32_t id, uint32_t ie);
} // namespace geom
