#ifndef VELOCITY_MSL
#define VELOCITY_MSL

#include "ScreenSpace.metal"

// Screen motion is stored at 1/100 scale so extreme projections stay within half-float range.
inline float4 PackVelocity(float4 motion) { return motion * 0.01f; }
inline float4 UnpackVelocity(float4 motion) { return motion * 100.0f; }

// Packs shutter-open-to-current and reversed current-to-close UV motion.
inline float4 PackScreenMotion(float2 prev_ndc, float2 curr_ndc, float2 next_ndc) {
    return PackVelocity(float4(ndc_to_uv_delta(prev_ndc - curr_ndc), ndc_to_uv_delta(curr_ndc - next_ndc)));
}

#endif
