// Adapted from KhronosGroup/glTF-Sample-Renderer (tonemapping.glsl), pulled 2026-02-16.

#ifndef TONEMAPPING_GLSL
#define TONEMAPPING_GLSL

vec3 linearTosRGB(vec3 color) {
    const vec3 cutoff = step(vec3(0.0031308), color);
    const vec3 lower = color * 12.92;
    const vec3 higher = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lower, higher, cutoff);
}

vec3 toneMapPBRNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    const float x = min(color.r, min(color.g, color.b));
    const float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    const float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1.0 - startCompression;
    const float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    const float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

#endif
