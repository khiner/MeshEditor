#ifndef SCREENSPACE_MSL
#define SCREENSPACE_MSL

#include <metal_stdlib>
using namespace metal;

// Convert positive-Y-up clip coordinates to top-down framebuffer coordinates.
inline float2 ndc_to_uv(float2 ndc) { return ndc * float2(0.5f, -0.5f) + 0.5f; }

// Converts an NDC displacement to the corresponding UV displacement without an origin shift.
inline float2 ndc_to_uv_delta(float2 ndc_delta) { return ndc_delta * float2(0.5f, -0.5f); }

#endif
