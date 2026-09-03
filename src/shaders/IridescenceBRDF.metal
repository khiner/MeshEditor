// Adapted from KhronosGroup/glTF-Sample-Renderer (iridescence.glsl)
// Implements thin-film interference (Belcour 2017).
// https://belcour.github.io/blog/research/2017/05/01/brdf-thin-film.html

#ifndef IRIDESCENCE_BRDF_MSL
#define IRIDESCENCE_BRDF_MSL

#include "brdf.metal"

inline float sq(float x) { return x * x; }
inline float3  sq(float3  x) { return x * x; }

constant float3x3 XYZ_TO_REC709 = float3x3(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

inline float3 Fresnel0ToIor(float3 fres) {
    const float3 sqrt_f0 = sqrt(fres);
    return (float3(1.0) + sqrt_f0) / (float3(1.0) - sqrt_f0);
}

inline float3 IorToFresnel0(float3 transmitted_ior, float incident_ior) {
    return sq((transmitted_ior - float3(incident_ior)) / (transmitted_ior + float3(incident_ior)));
}

inline float3 evalSensitivity(float opd, float3 shift) {
    const float phase = 2.0 * Pi * opd * 1.0e-9;
    const float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    const float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    const float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    float3 xyz = val * sqrt(2.0 * Pi * var) * cos(pos * phase + shift) * exp(-sq(phase) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * Pi * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * sq(phase));
    return XYZ_TO_REC709 * (xyz / 1.0685e-7);
}

// Evaluates two-interface thin-film Fresnel reflectance with thickness in nanometers.
inline float3 evalIridescence(float outsideIOR, float eta2, float cos_theta_1, float thinFilmThickness, float3 baseF0) {
    // Approach the outside IOR continuously as thickness approaches zero.
    float iridescence_ior = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));

    // Total internal reflection produces unit reflectance.
    float cos_theta_2_sq = 1.0 - sq(outsideIOR / iridescence_ior) * (1.0 - sq(cos_theta_1));
    if (cos_theta_2_sq < 0.0) return float3(1.0);
    float cos_theta_2 = sqrt(cos_theta_2_sq);

    float r0 = sq((iridescence_ior - outsideIOR) / (iridescence_ior + outsideIOR));
    float r12 = r0 + (1.0 - r0) * pow(clamp(1.0 - cos_theta_1, 0.0, 1.0), 5.0);
    float t121 = 1.0 - r12;
    float phi12 = 0.0;
    if (iridescence_ior < outsideIOR) phi12 = Pi;
    float phi21 = Pi - phi12;

    const float3 base_ior = Fresnel0ToIor(clamp(baseF0, 0.0, 0.9999));
    const float3 r1 = IorToFresnel0(base_ior, iridescence_ior);
    const float3 r23 = F_Schlick(r1, float3(1.0), cos_theta_2);
    float3 phi23 = float3(0.0);
    if (base_ior[0] < iridescence_ior) phi23[0] = Pi;
    if (base_ior[1] < iridescence_ior) phi23[1] = Pi;
    if (base_ior[2] < iridescence_ior) phi23[2] = Pi;

    const float opd = 2.0 * iridescence_ior * thinFilmThickness * cos_theta_2;
    const float3 phi = float3(phi21) + phi23;

    const float3 r123_sq = clamp(r12 * r23, 1e-5, 0.9999);
    const float3 r123 = sqrt(r123_sq);
    const float3 Rs = sq(t121) * r23 / (float3(1.0) - r123_sq);

    float3 I = r12 + Rs;
    float3 Cm = Rs - t121;
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        I += Cm * 2.0 * evalSensitivity(float(m) * opd, float(m) * phi);
    }

    return max(I, float3(0.0));
}

#endif
