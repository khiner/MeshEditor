#ifndef SCREENSPACE_MSL
#define SCREENSPACE_MSL

#include <metal_stdlib>
using namespace metal;

// Clip space runs y up from -1 at the bottom of the viewport, while framebuffer rows and texture V
// run top-down, so a y crossing between them mirrors. Every render target holds its image upright:
// row 0 is the top of the screen.
inline float2 ndc_to_uv(float2 ndc) { return ndc * float2(0.5f, -0.5f) + 0.5f; }

// A difference of NDC positions carries the same mirror without the origin shift, so this is
// `ndc_to_uv(a) - ndc_to_uv(b)` for `a - b`.
inline float2 ndc_to_uv_delta(float2 ndc_delta) { return ndc_delta * float2(0.5f, -0.5f); }

#endif
