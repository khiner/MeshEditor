#ifndef VELOCITY_MSL
#define VELOCITY_MSL

#include "ScreenSpace.metal"

// Screen motion is stored at 1/100 scale so extreme projections stay within half-float range.
inline float4 PackVelocity(float4 motion) { return motion * 0.01f; }
inline float4 UnpackVelocity(float4 motion) { return motion * 100.0f; }

// A surface's screen motion across the shutter, as the UV step into this frame and the step out of
// it. The second half is stored pointing backward like the first, which the gather's motion scale undoes.
inline float4 PackScreenMotion(float2 prev_ndc, float2 curr_ndc, float2 next_ndc) {
    return PackVelocity(float4(ndc_to_uv_delta(prev_ndc - curr_ndc), ndc_to_uv_delta(curr_ndc - next_ndc)));
}

#endif
