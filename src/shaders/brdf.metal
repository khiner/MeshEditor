#pragma once
// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl)
#include <metal_stdlib>
using namespace metal;

constant float Pi = 3.14159265358979323846f;

inline float3 F_Schlick(float3 f0, float3 f90, float VdotH) { return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0); }

inline float V_GGX(float NdotL, float NdotV, float alpha_roughness) {
    const float alpha_roughness_sq = alpha_roughness * alpha_roughness;

    const float ggxv = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha_roughness_sq) + alpha_roughness_sq);
    const float ggxl = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha_roughness_sq) + alpha_roughness_sq);
    const float ggx = ggxv + ggxl;
    if (ggx > 0.0) return 0.5 / ggx;
    return 0.0;
}

inline float D_GGX(float NdotH, float alpha_roughness) {
    const float alpha_roughness_sq = alpha_roughness * alpha_roughness;
    const float f = (NdotH * NdotH) * (alpha_roughness_sq - 1.0) + 1.0;
    return alpha_roughness_sq / (Pi * f * f);
}

inline float D_GGX_anisotropic(float NdotH, float TdotH, float BdotH, float at, float ab) {
    const float a2 = at * ab;
    const float3 f = float3(ab * TdotH, at * BdotH, a2 * NdotH);
    const float w2 = a2 / dot(f, f);
    return a2 * w2 * w2 / Pi;
}

inline float V_GGX_anisotropic(float NdotL, float NdotV, float BdotV, float TdotV, float TdotL, float BdotL, float at, float ab) {
    const float ggxv = NdotL * length(float3(at * TdotV, ab * BdotV, NdotV));
    const float ggxl = NdotV * length(float3(at * TdotL, ab * BdotL, NdotL));
    return clamp(0.5 / (ggxv + ggxl), 0.0, 1.0);
}

inline float3 BRDF_specularGGXAnisotropy(float alpha_roughness, float anisotropy, float3 n, float3 v, float3 l, float3 h, float3 t, float3 b) {
    const float at = mix(alpha_roughness, 1.0, anisotropy * anisotropy);
    const float ab = clamp(alpha_roughness, 0.001, 1.0);
    const float NdotL = clamp(dot(n, l), 0.0, 1.0);
    const float NdotV = dot(n, v);
    const float NdotH = clamp(dot(n, h), 0.001, 1.0);
    return float3(
        V_GGX_anisotropic(NdotL, NdotV, dot(b, v), dot(t, v), dot(t, l), dot(b, l), at, ab) * D_GGX_anisotropic(NdotH, dot(t, h), dot(b, h), at, ab)
    );
}

