// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl), pulled 2026-02-16.

#ifndef BRDF_GLSL
#define BRDF_GLSL

const float M_PI = 3.14159265358979323846;

vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH) {
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float V_GGX(float NdotL, float NdotV, float alphaRoughness) {
    const float alpha_roughness_sq = alphaRoughness * alphaRoughness;

    const float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha_roughness_sq) + alpha_roughness_sq);
    const float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha_roughness_sq) + alpha_roughness_sq);

    const float GGX = GGXV + GGXL;
    if (GGX > 0.0) return 0.5 / GGX;
    return 0.0;
}

float D_GGX(float NdotH, float alphaRoughness) {
    const float alpha_roughness_sq = alphaRoughness * alphaRoughness;
    const float f = (NdotH * NdotH) * (alpha_roughness_sq - 1.0) + 1.0;
    return alpha_roughness_sq / (M_PI * f * f);
}

vec3 BRDF_lambertian(vec3 diffuseColor) {
    return diffuseColor / M_PI;
}

vec3 BRDF_specularGGX(float alphaRoughness, float NdotL, float NdotV, float NdotH) {
    const float V = V_GGX(NdotL, NdotV, alphaRoughness);
    const float D = D_GGX(NdotH, alphaRoughness);
    return vec3(V * D);
}

#endif
