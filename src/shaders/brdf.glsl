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

// KHR_materials_anisotropy — anisotropic GGX BRDF.
// Adapted from KhronosGroup/glTF-Sample-Renderer (brdf.glsl).
float D_GGX_anisotropic(float NdotH, float TdotH, float BdotH, float at, float ab) {
    const float a2 = at * ab;
    const vec3 f = vec3(ab * TdotH, at * BdotH, a2 * NdotH);
    const float w2 = a2 / dot(f, f);
    return a2 * w2 * w2 / M_PI;
}

float V_GGX_anisotropic(float NdotL, float NdotV, float BdotV, float TdotV, float TdotL, float BdotL, float at, float ab) {
    const float GGXV = NdotL * length(vec3(at * TdotV, ab * BdotV, NdotV));
    const float GGXL = NdotV * length(vec3(at * TdotL, ab * BdotL, NdotL));
    return clamp(0.5 / (GGXV + GGXL), 0.0, 1.0);
}

vec3 BRDF_specularGGXAnisotropy(float alphaRoughness, float anisotropy, vec3 n, vec3 v, vec3 l, vec3 h, vec3 t, vec3 b) {
    const float at = mix(alphaRoughness, 1.0, anisotropy * anisotropy);
    const float ab = clamp(alphaRoughness, 0.001, 1.0);
    const float NdotL = clamp(dot(n, l), 0.0, 1.0);
    const float NdotV = dot(n, v);
    const float NdotH = clamp(dot(n, h), 0.001, 1.0);
    const float V = V_GGX_anisotropic(NdotL, NdotV, dot(b, v), dot(t, v), dot(t, l), dot(b, l), at, ab);
    const float D = D_GGX_anisotropic(NdotH, dot(t, h), dot(b, h), at, ab);
    return vec3(V * D);
}

// KHR_materials_sheen — Charlie sheen BRDF.
// Estevez and Kulla, "Production Friendly Microfacet Sheen BRDF", SIGGRAPH 2017.
float lambdaSheenNumericHelper(float x, float alphaG) {
    const float oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
    const float a = mix(21.5473, 25.3245, oneMinusAlphaSq);
    const float b = mix(3.82987, 3.32435, oneMinusAlphaSq);
    const float c = mix(0.19823, 0.16801, oneMinusAlphaSq);
    const float d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
    const float e = mix(-4.32054, -4.85967, oneMinusAlphaSq);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float lambdaSheen(float cosTheta, float alphaG) {
    if (abs(cosTheta) < 0.5) return exp(lambdaSheenNumericHelper(cosTheta, alphaG));
    return exp(2.0 * lambdaSheenNumericHelper(0.5, alphaG) - lambdaSheenNumericHelper(1.0 - cosTheta, alphaG));
}

float V_Sheen(float NdotL, float NdotV, float sheenRoughness) {
    sheenRoughness = max(sheenRoughness, 0.000001);
    const float alphaG = sheenRoughness * sheenRoughness;
    return clamp(1.0 / ((1.0 + lambdaSheen(NdotV, alphaG) + lambdaSheen(NdotL, alphaG)) * (4.0 * NdotV * NdotL)), 0.0, 1.0);
}

float D_Charlie(float sheenRoughness, float NdotH) {
    sheenRoughness = max(sheenRoughness, 0.000001);
    const float alphaG = sheenRoughness * sheenRoughness;
    const float invR = 1.0 / alphaG;
    const float cos2h = NdotH * NdotH;
    const float sin2h = 1.0 - cos2h;
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * M_PI);
}

vec3 BRDF_specularSheen(vec3 sheenColor, float sheenRoughness, float NdotL, float NdotV, float NdotH) {
    return sheenColor * D_Charlie(sheenRoughness, NdotH) * V_Sheen(NdotL, NdotV, sheenRoughness);
}

#endif
