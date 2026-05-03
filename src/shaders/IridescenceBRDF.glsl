// Adapted from KhronosGroup/glTF-Sample-Renderer (iridescence.glsl)
// Implements thin-film interference (Belcour 2017).
// https://belcour.github.io/blog/research/2017/05/01/brdf-thin-film.html

#ifndef IRIDESCENCE_BRDF_GLSL
#define IRIDESCENCE_BRDF_GLSL

float sq(float x) { return x * x; }
vec3  sq(vec3  x) { return x * x; }

// XYZ to sRGB color space matrix.
const mat3 XYZ_TO_REC709 = mat3(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

vec3 Fresnel0ToIor(vec3 fres) {
    const vec3 sqrt_f0 = sqrt(fres);
    return (vec3(1.0) + sqrt_f0) / (vec3(1.0) - sqrt_f0);
}

vec3 IorToFresnel0(vec3 transmitted_ior, float incident_ior) {
    return sq((transmitted_ior - vec3(incident_ior)) / (transmitted_ior + vec3(incident_ior)));
}

// Evaluation of XYZ sensitivity curves in Fourier space.
vec3 evalSensitivity(float opd, vec3 shift) {
    const float phase = 2.0 * M_PI * opd * 1.0e-9;
    const vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    const vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    const vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    vec3 xyz = val * sqrt(2.0 * M_PI * var) * cos(pos * phase + shift) * exp(-sq(phase) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * M_PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * sq(phase));
    return XYZ_TO_REC709 * (xyz / 1.0685e-7);
}

// Thin-film interference Fresnel (two-interface model: air -> film -> base).
// outsideIOR: IOR of the medium above the film (1.0 for air).
// eta2:       IOR of the thin film (iridescence_ior).
// cos_theta_1:  cosine of the angle of incidence on the outer surface.
// thinFilmThickness: film thickness in nanometers.
// baseF0:     base material F0 (per-channel).
vec3 evalIridescence(float outsideIOR, float eta2, float cos_theta_1, float thinFilmThickness, vec3 baseF0) {
    // Smoothly fade iridescence_ior -> outsideIOR as thickness -> 0 to avoid a hard cutoff.
    float iridescence_ior = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));

    // Total internal reflection: no transmission, return opaque white.
    float cos_theta_2_sq = 1.0 - sq(outsideIOR / iridescence_ior) * (1.0 - sq(cos_theta_1));
    if (cos_theta_2_sq < 0.0) return vec3(1.0);
    float cos_theta_2 = sqrt(cos_theta_2_sq);

    // First interface (air <-> film) — scalar Fresnel.
    float r0 = sq((iridescence_ior - outsideIOR) / (iridescence_ior + outsideIOR));
    float r12 = r0 + (1.0 - r0) * pow(clamp(1.0 - cos_theta_1, 0.0, 1.0), 5.0);
    float t121 = 1.0 - r12;
    float phi12 = 0.0;
    if (iridescence_ior < outsideIOR) phi12 = M_PI;
    float phi21 = M_PI - phi12;

    // Second interface (film <-> base) — per-channel Fresnel.
    const vec3 base_ior = Fresnel0ToIor(clamp(baseF0, 0.0, 0.9999)); // guard against F0=1
    const vec3 r1 = IorToFresnel0(base_ior, iridescence_ior);
    const vec3 r23 = F_Schlick(r1, vec3(1.0), cos_theta_2);
    vec3 phi23 = vec3(0.0);
    if (base_ior[0] < iridescence_ior) phi23[0] = M_PI;
    if (base_ior[1] < iridescence_ior) phi23[1] = M_PI;
    if (base_ior[2] < iridescence_ior) phi23[2] = M_PI;

    // Optical path difference and total phase shift.
    const float opd = 2.0 * iridescence_ior * thinFilmThickness * cos_theta_2;
    const vec3 phi = vec3(phi21) + phi23;

    // Compound reflectance terms.
    const vec3 r123_sq = clamp(r12 * r23, 1e-5, 0.9999);
    const vec3 r123 = sqrt(r123_sq);
    const vec3 Rs = sq(t121) * r23 / (vec3(1.0) - r123_sq);

    // m=0 (DC) term.
    vec3 I = r12 + Rs;
    // m>0 (pairs of diracs): two interference orders.
    vec3 Cm = Rs - t121;
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        I += Cm * 2.0 * evalSensitivity(float(m) * opd, float(m) * phi);
    }

    return max(I, vec3(0.0));
}

#endif
