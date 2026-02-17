// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl), pulled 2026-02-16.

#ifndef BRDF_GLSL
#define BRDF_GLSL

const float M_PI = 3.14159265358979323846;

vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH) {
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float V_GGX(float NdotL, float NdotV, float alphaRoughness) {
    const float alphaSq = alphaRoughness * alphaRoughness;
    const float GGXV = NdotL * sqrt(max(NdotV * NdotV * (1.0 - alphaSq) + alphaSq, 0.0));
    const float GGXL = NdotV * sqrt(max(NdotL * NdotL * (1.0 - alphaSq) + alphaSq, 0.0));
    const float GGX = GGXV + GGXL;
    return GGX > 0.0 ? 0.5 / GGX : 0.0;
}

float D_GGX(float NdotH, float alphaRoughness) {
    const float alphaSq = alphaRoughness * alphaRoughness;
    const float f = (NdotH * NdotH) * (alphaSq - 1.0) + 1.0;
    return alphaSq / (M_PI * f * f);
}

vec3 BRDF_lambertian(vec3 diffuseColor) {
    return diffuseColor / M_PI;
}

vec3 BRDF_specularGGX(vec3 f0, vec3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH) {
    const vec3 F = F_Schlick(f0, f90, VdotH);
    const float V = V_GGX(NdotL, NdotV, alphaRoughness);
    const float D = D_GGX(NdotH, alphaRoughness);
    return F * (V * D);
}

#endif
