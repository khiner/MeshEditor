// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl)

#ifndef BRDF_MSL
#define BRDF_MSL

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

inline float3 BRDF_lambertian(float3 diffuse_color) { return diffuse_color / Pi; }

inline float3 BRDF_specularGGX(float alpha_roughness, float NdotL, float NdotV, float NdotH) {
    return float3(V_GGX(NdotL, NdotV, alpha_roughness) * D_GGX(NdotH, alpha_roughness));
}

// KHR_materials_anisotropy — anisotropic GGX BRDF.
// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl).
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

// KHR_materials_sheen — Charlie sheen BRDF.
// Estevez and Kulla, "Production Friendly Microfacet Sheen BRDF", SIGGRAPH 2017.
inline float lambdaSheenNumericHelper(float x, float alpha_g) {
    const float one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    const float a = mix(21.5473, 25.3245, one_minus_alpha_sq);
    const float b = mix(3.82987, 3.32435, one_minus_alpha_sq);
    const float c = mix(0.19823, 0.16801, one_minus_alpha_sq);
    const float d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
    const float e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
}

inline float lambdaSheen(float cos_theta, float alpha_g) {
    if (abs(cos_theta) < 0.5) return exp(lambdaSheenNumericHelper(cos_theta, alpha_g));
    return exp(2.0 * lambdaSheenNumericHelper(0.5, alpha_g) - lambdaSheenNumericHelper(1.0 - cos_theta, alpha_g));
}

inline float V_Sheen(float NdotL, float NdotV, float sheen_roughness) {
    sheen_roughness = max(sheen_roughness, 0.000001);
    const float alpha_g = sheen_roughness * sheen_roughness;
    return clamp(1.0 / ((1.0 + lambdaSheen(NdotV, alpha_g) + lambdaSheen(NdotL, alpha_g)) * (4.0 * NdotV * NdotL)), 0.0, 1.0);
}

inline float D_Charlie(float sheen_roughness, float NdotH) {
    sheen_roughness = max(sheen_roughness, 0.000001);
    const float alpha_g = sheen_roughness * sheen_roughness;
    const float inv_r = 1.0 / alpha_g;
    return (2.0 + inv_r) * pow(1.0 - NdotH * NdotH, inv_r * 0.5) / (2.0 * Pi);
}

inline float3 BRDF_specularSheen(float3 sheen_color, float sheen_roughness, float NdotL, float NdotV, float NdotH) {
    return sheen_color * D_Charlie(sheen_roughness, NdotH) * V_Sheen(NdotL, NdotV, sheen_roughness);
}

#endif
