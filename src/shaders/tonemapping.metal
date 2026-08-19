// Adapted from KhronosGroup/glTF-Sample-Renderer (tonemapping.glsl)

#ifndef TONEMAPPING_MSL
#define TONEMAPPING_MSL

#include <metal_stdlib>
using namespace metal;

constant float GAMMA = 2.2f;
constant float INV_GAMMA = 1.0f / GAMMA;

inline float3 linearTosRGB(float3 color) { return pow(color, float3(INV_GAMMA)); }
inline float3 sRGBToLinear(float3 color) { return pow(color, float3(GAMMA)); }

inline float3 toneMapPBRNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    const float x = min(color.r, min(color.g, color.b));
    const float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    const float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1.0f - startCompression;
    const float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    const float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return mix(color, float3(newPeak), g);
}

// Display transform: tone map, then sRGB encode.
inline float3 linearToDisplay(float3 color) { return linearTosRGB(toneMapPBRNeutral(color)); }

#endif
