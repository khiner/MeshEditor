// Adapted from KhronosGroup/glTF-Sample-Renderer (iridescence.glsl), pulled 2026-02-24.
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

vec3 Fresnel0ToIor(vec3 fresnel0) {
    const vec3 sqrtF0 = sqrt(fresnel0);
    return (vec3(1.0) + sqrtF0) / (vec3(1.0) - sqrtF0);
}

vec3 IorToFresnel0(vec3 transmittedIor, float incidentIor) {
    return sq((transmittedIor - vec3(incidentIor)) / (transmittedIor + vec3(incidentIor)));
}

// Evaluation of XYZ sensitivity curves in Fourier space.
vec3 evalSensitivity(float OPD, vec3 shift) {
    const float phase = 2.0 * M_PI * OPD * 1.0e-9;
    const vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    const vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    const vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    vec3 xyz = val * sqrt(2.0 * M_PI * var) * cos(pos * phase + shift) * exp(-sq(phase) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * M_PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * sq(phase));
    xyz /= 1.0685e-7;

    return XYZ_TO_REC709 * xyz;
}

// Thin-film interference Fresnel (two-interface model: air -> film -> base).
// outsideIOR: IOR of the medium above the film (1.0 for air).
// eta2:       IOR of the thin film (iridescenceIor).
// cosTheta1:  cosine of the angle of incidence on the outer surface.
// thinFilmThickness: film thickness in nanometers.
// baseF0:     base material F0 (per-channel).
vec3 evalIridescence(float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0) {
    // Smoothly fade iridescenceIor -> outsideIOR as thickness -> 0 to avoid a hard cutoff.
    float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    float sinTheta2Sq = sq(outsideIOR / iridescenceIor) * (1.0 - sq(cosTheta1));

    // Total internal reflection: no transmission, return opaque white.
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) return vec3(1.0);
    float cosTheta2 = sqrt(cosTheta2Sq);

    // First interface (air <-> film) — scalar Fresnel.
    float R0 = sq((iridescenceIor - outsideIOR) / (iridescenceIor + outsideIOR));
    float R12 = R0 + (1.0 - R0) * pow(clamp(1.0 - cosTheta1, 0.0, 1.0), 5.0);
    float T121 = 1.0 - R12;
    float phi12 = 0.0;
    if (iridescenceIor < outsideIOR) phi12 = M_PI;
    float phi21 = M_PI - phi12;

    // Second interface (film <-> base) — per-channel Fresnel.
    const vec3 baseIOR = Fresnel0ToIor(clamp(baseF0, 0.0, 0.9999)); // guard against F0=1
    const vec3 R1 = IorToFresnel0(baseIOR, iridescenceIor);
    const vec3 R23 = F_Schlick(R1, vec3(1.0), cosTheta2);
    vec3 phi23 = vec3(0.0);
    if (baseIOR[0] < iridescenceIor) phi23[0] = M_PI;
    if (baseIOR[1] < iridescenceIor) phi23[1] = M_PI;
    if (baseIOR[2] < iridescenceIor) phi23[2] = M_PI;

    // Optical path difference and total phase shift.
    const float OPD = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    const vec3 phi = vec3(phi21) + phi23;

    // Compound reflectance terms.
    const vec3 R123 = clamp(R12 * R23, 1e-5, 0.9999);
    const vec3 r123 = sqrt(R123);
    const vec3 Rs = sq(T121) * R23 / (vec3(1.0) - R123);

    // m=0 (DC) term.
    vec3 I = R12 + Rs;

    // m>0 (pairs of diracs): two interference orders.
    vec3 Cm = Rs - T121;
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        I += Cm * 2.0 * evalSensitivity(float(m) * OPD, float(m) * phi);
    }

    return max(I, vec3(0.0));
}

#endif
