// Adapted from KhronosGroup/glTF-Sample-Renderer (tonemapping.glsl)

#ifndef TONEMAPPING_GLSL
#define TONEMAPPING_GLSL

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

vec3 linearTosRGB(vec3 color) {
    return pow(color, vec3(INV_GAMMA));
}

vec3 sRGBToLinear(vec3 color) {
    return pow(color, vec3(GAMMA));
}

// DisplayToSceneLinear in ViewportRenderGpu.cpp is the CPU inverse of the display transform
// (toneMapPBRNeutral then linearTosRGB) and must track changes here.
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

// Display transform: tone map, then sRGB encode.
vec3 linearToDisplay(vec3 color) {
    return linearTosRGB(toneMapPBRNeutral(color));
}

// Inverse of linearToDisplay, ignoring the tone map's non-invertible desaturation term (negligible near the
// knee). The CPU twin is DisplayToSceneLinear in ViewportRenderGpu.cpp; keep the two in sync.
vec3 displayToLinear(vec3 color) {
    color = sRGBToLinear(color);
    const float startCompression = 0.8 - 0.04;
    const float d = 1.0 - startCompression;
    // The forward compression asymptotes to 1, so clamp just below to keep the inverse finite.
    const float peak = min(max(color.r, max(color.g, color.b)), 0.999);
    if (peak >= startCompression) {
        const float originalPeak = d * d / (1.0 - peak) - d + startCompression;
        color *= originalPeak / peak;
    }
    // Undo the black-level offset. The forward maps min channel x to 6.25x^2 for x < 0.08, else x - 0.04.
    const float x = min(color.r, min(color.g, color.b));
    const float offset = x < 0.04 ? 0.4 * sqrt(x) - x : 0.04;
    return color + offset;
}

#endif
