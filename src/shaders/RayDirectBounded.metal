// Fast math that keeps Inf and NaN semantics.
#pragma METAL fp math_mode(relaxed)
// The general RGB path stays available in this kernel for receivers inside the caster footprint.
#define BOUNDED_AREA_SHADOWS
#include "RayLighting.metal"
