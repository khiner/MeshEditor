// Build the one-ray SIMD kernel independently so its register layout cannot
// change the glass and multi-ray intersection kernels.
#define FILTER_AREA_SHADOWS
#include "RayLighting.metal"
