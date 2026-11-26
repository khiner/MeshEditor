#pragma once
#include "AreaLight.h"
#include "EeveeDiffuse.h"
#include "EeveeGlass.h"
#include "IridescenceBRDF.metal"
#include "ibl.metal"

// The lobe types of a glTF surface.
// Each type defines its value for one light direction (the BSDF times the cosine), its directional albedo, its prefiltered environment, its area-light integral, its sample and its probability.
// A closure scales each lobe by a weight that depends on the view direction only.

// GGX microfacet reflection with separable Smith shadowing and a Schlick Fresnel term from F0 at normal incidence to F90 at grazing incidence.
inline float GgxAlpha(float roughness) { return max(roughness * roughness, 1e-3f); }

// The value without its Fresnel factor.
inline float GgxValue(float alpha, float NdotL, float NdotV, float NdotH) {
    using namespace eevee_microfacet;
    const float a2 = alpha * alpha;
    return NdotL > 0.0f ? bxdf_ggx_D(max(NdotH, 1e-8f), a2) * bxdf_ggx_smith_G1(max(NdotV, 1e-8f), a2) * bxdf_ggx_smith_G1(NdotL, a2) / (4.0f * max(NdotV, 1e-8f)) : 0.0f;
}
inline float GgxAnisotropicValue(float alpha, float anisotropy, float3 n, float3 v, float3 l, float3 h, float3 t, float3 b, float NdotL) {
    return NdotL * BRDF_specularGGXAnisotropy(alpha, anisotropy, n, v, l, h, t, b).x;
}

// The single-scattering directional albedo as weights of F0 and F90.
inline float2 GgxAlbedoTerms(texture2d_array<float> lut, float roughness, float NdotV) {
    return eevee_glass::lut2(lut, float2(roughness, sqrt(saturate(1.0f - NdotV))), 1u).xy;
}
// Scale that adds the light scattered more than once between microfacets, from the hemispherical average of the Fresnel term.
inline float3 GgxCompensation(float3 f0, float3 f90, float2 terms) {
    return 1.0f / (1.0f - (1.0f - terms.x - terms.y) * (f0 + (f90 - f0) / 21.0f));
}

inline float3 GgxEnvironment(const thread Scene &scene, float3 n, float3 v, float roughness) { return getIBLRadianceGGX(scene, n, v, roughness); }
inline float GgxArea(texture2d_array<float> lut, const thread AreaEmitter &emitter, eevee_ltc::LTCData ltc) {
    return AreaIntegral(lut, emitter.Shape, normalize(emitter.Center), emitter.Rectangle, ltc);
}

// Samples the visible normals of the bounded cap above the view direction.
// `u.x` selects the height in the cap and `u.y` the angle around the normal.
// A reflection below the surface has zero probability.
inline eevee_microfacet::BsdfSample GgxSample(float3 n, float3 v, float alpha, float2 u) {
    const float3x3 frame = RayFrame(n);
    const float phi = 2.0f * M_PI_F * u.y;
    auto sample = eevee_microfacet::bxdf_ggx_sample_reflection(float3(u.x, cos(phi), sin(phi)), transpose(frame) * v, alpha, true);
    sample.direction = frame * sample.direction;
    if (dot(n, sample.direction) <= 0.0f) sample.pdf = 0.0f;
    return sample;
}
inline float GgxPdf(float3 n, float3 v, float3 l, float alpha) { return dot(n, l) > 0.0f ? eevee_microfacet::bxdf_ggx_eval_reflection(n, l, v, alpha, true).pdf : 0.0f; }

// Lambertian reflection about `n`, which is transmission about the opposite normal.
inline float DiffuseValue(float3 n, float3 l) { return saturate(dot(n, l)) / M_PI_F; }
inline float3 DiffuseEnvironment(const thread Scene &scene, float3 n) { return getDiffuseLight(scene, n); }
inline float DiffuseArea(texture2d_array<float> lut, const thread AreaEmitter &emitter, eevee_ltc::LTCData ltc) {
    return AreaIntegral(lut, emitter.Shape, normalize(emitter.Center), emitter.Rectangle, ltc);
}
// Samples the cosine-weighted hemisphere about `n`, or about `-n` for transmission.
inline float3 DiffuseSample(float3 n, float2 u, bool transmission) { return RayFrame(n) * eevee_diffuse::sample(u, transmission); }
// The probability of the mixture that transmits a `transmission` fraction of its samples.
inline float DiffusePdf(float3 n, float3 l, float transmission) { return eevee_diffuse::pdf(dot(n, l), transmission); }

// Sheen: the Charlie distribution and its sheen visibility term, whose `alpha` is the squared sheen roughness.
// Sheen scatters over the whole hemisphere, so its prefiltered environment and its area-light integral are the diffuse ones.
inline float SheenAlpha(float roughness) { return max(roughness * roughness, 1e-12f); }
inline float SheenLambda(float cosine, float alpha) {
    const float s = (1.0f - alpha) * (1.0f - alpha);
    const float a = mix(21.5473f, 25.3245f, s), b = mix(3.82987f, 3.32435f, s), c = mix(0.19823f, 0.16801f, s), d = mix(-1.97760f, -1.27393f, s), e = mix(-4.32054f, -4.85967f, s);
    const auto l = [&](float x) { return a / (1.0f + b * pow(x, c)) + d * x + e; };
    return exp(abs(cosine) < 0.5f ? l(cosine) : 2.0f * l(0.5f) - l(1.0f - cosine));
}
inline float SheenValue(float alpha, float NdotL, float NdotV, float NdotH) {
    const float inverse = 1.0f / alpha;
    const float distribution = (2.0f + inverse) * pow(saturate(1.0f - NdotH * NdotH), inverse * 0.5f) / (2.0f * M_PI_F);
    const float visibility = saturate(1.0f / ((1.0f + SheenLambda(NdotV, alpha) + SheenLambda(NdotL, alpha)) * (4.0f * NdotV * NdotL)));
    return NdotL * distribution * visibility;
}
// Directional albedo over the square root of NdotV and the sheen roughness, integrated numerically.
constant float SheenAlbedoTable[32][32] = {
    {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    {1.9527f, 1.9269f, 1.7883f, 1.5662f, 1.2911f, 0.9955f, 0.7113f, 0.4654f, 0.2745f, 0.1429f, 0.0638f, 0.0235f, 0.0067f, 0.0015f, 0.0003f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    {1.6302f, 1.6954f, 1.6721f, 1.5741f, 1.4289f, 1.2523f, 1.0580f, 0.8587f, 0.6664f, 0.4912f, 0.3411f, 0.2210f, 0.1325f, 0.0740f, 0.0384f, 0.0182f, 0.0078f, 0.0030f, 0.0010f, 0.0003f, 0.0001f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    {1.3256f, 1.4157f, 1.4341f, 1.3847f, 1.2955f, 1.1796f, 1.0459f, 0.9021f, 0.7553f, 0.6122f, 0.4786f, 0.3597f, 0.2594f, 0.1803f, 0.1207f, 0.0774f, 0.0473f, 0.0273f, 0.0148f, 0.0075f, 0.0035f, 0.0014f, 0.0005f, 0.0002f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    {1.1105f, 1.2035f, 1.2371f, 1.2107f, 1.1506f, 1.0682f, 0.9706f, 0.8630f, 0.7504f, 0.6371f, 0.5273f, 0.4248f, 0.3330f, 0.2548f, 0.1904f, 0.1384f, 0.0977f, 0.0666f, 0.0437f, 0.0274f, 0.0163f, 0.0092f, 0.0048f, 0.0023f, 0.0010f, 0.0004f, 0.0001f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    {0.9599f, 1.0499f, 1.0893f, 1.0754f, 1.0323f, 0.9703f, 0.8953f, 0.8114f, 0.7222f, 0.6308f, 0.5402f, 0.4534f, 0.3729f, 0.3012f, 0.2392f, 0.1863f, 0.1421f, 0.1058f, 0.0767f, 0.0539f, 0.0366f, 0.0238f, 0.0148f, 0.0087f, 0.0048f, 0.0024f, 0.0011f, 0.0004f, 0.0001f, 0.0000f, 0.0000f, 0.0000f},
    {0.8518f, 0.9377f, 0.9795f, 0.9730f, 0.9407f, 0.8921f, 0.8321f, 0.7643f, 0.6913f, 0.6157f, 0.5397f, 0.4654f, 0.3950f, 0.3305f, 0.2728f, 0.2220f, 0.1778f, 0.1398f, 0.1077f, 0.0811f, 0.0595f, 0.0424f, 0.0291f, 0.0192f, 0.0121f, 0.0072f, 0.0039f, 0.0019f, 0.0008f, 0.0003f, 0.0001f, 0.0000f},
    {0.7725f, 0.8545f, 0.8972f, 0.8955f, 0.8707f, 0.8312f, 0.7817f, 0.7252f, 0.6640f, 0.5999f, 0.5348f, 0.4703f, 0.4082f, 0.3500f, 0.2970f, 0.2490f, 0.2062f, 0.1683f, 0.1352f, 0.1066f, 0.0824f, 0.0622f, 0.0456f, 0.0325f, 0.0223f, 0.0146f, 0.0090f, 0.0051f, 0.0026f, 0.0011f, 0.0003f, 0.0000f},
    {0.7135f, 0.7922f, 0.8352f, 0.8369f, 0.8173f, 0.7844f, 0.7426f, 0.6944f, 0.6417f, 0.5863f, 0.5295f, 0.4726f, 0.4172f, 0.3644f, 0.3154f, 0.2704f, 0.2295f, 0.1924f, 0.1592f, 0.1298f, 0.1041f, 0.0819f, 0.0629f, 0.0472f, 0.0345f, 0.0242f, 0.0163f, 0.0102f, 0.0059f, 0.0029f, 0.0011f, 0.0002f},
    {0.6691f, 0.7451f, 0.7884f, 0.7925f, 0.7768f, 0.7489f, 0.7127f, 0.6708f, 0.6247f, 0.5758f, 0.5254f, 0.4746f, 0.4244f, 0.3761f, 0.3306f, 0.2883f, 0.2493f, 0.2133f, 0.1806f, 0.1510f, 0.1244f, 0.1009f, 0.0803f, 0.0627f, 0.0478f, 0.0354f, 0.0252f, 0.0171f, 0.0107f, 0.0060f, 0.0027f, 0.0007f},
    {0.6358f, 0.7097f, 0.7531f, 0.7592f, 0.7464f, 0.7223f, 0.6906f, 0.6534f, 0.6124f, 0.5686f, 0.5233f, 0.4772f, 0.4313f, 0.3866f, 0.3441f, 0.3042f, 0.2668f, 0.2321f, 0.2000f, 0.1705f, 0.1435f, 0.1192f, 0.0974f, 0.0783f, 0.0618f, 0.0476f, 0.0355f, 0.0253f, 0.0170f, 0.0104f, 0.0054f, 0.0018f},
    {0.6110f, 0.6834f, 0.7270f, 0.7346f, 0.7242f, 0.7030f, 0.6747f, 0.6413f, 0.6043f, 0.5646f, 0.5232f, 0.4810f, 0.4386f, 0.3969f, 0.3568f, 0.3189f, 0.2831f, 0.2494f, 0.2180f, 0.1887f, 0.1617f, 0.1368f, 0.1142f, 0.0940f, 0.0761f, 0.0604f, 0.0467f, 0.0348f, 0.0246f, 0.0161f, 0.0092f, 0.0039f},
    {0.5930f, 0.6642f, 0.7081f, 0.7170f, 0.7085f, 0.6897f, 0.6641f, 0.6337f, 0.5999f, 0.5635f, 0.5253f, 0.4862f, 0.4466f, 0.4075f, 0.3694f, 0.3330f, 0.2985f, 0.2659f, 0.2350f, 0.2061f, 0.1790f, 0.1538f, 0.1306f, 0.1095f, 0.0907f, 0.0738f, 0.0587f, 0.0452f, 0.0334f, 0.0231f, 0.0143f, 0.0070f},
    {0.5806f, 0.6508f, 0.6952f, 0.7052f, 0.6982f, 0.6813f, 0.6579f, 0.6300f, 0.5987f, 0.5649f, 0.5294f, 0.4928f, 0.4556f, 0.4185f, 0.3821f, 0.3470f, 0.3136f, 0.2818f, 0.2515f, 0.2229f, 0.1958f, 0.1704f, 0.1467f, 0.1250f, 0.1053f, 0.0874f, 0.0711f, 0.0564f, 0.0430f, 0.0310f, 0.0205f, 0.0113f},
    {0.5726f, 0.6423f, 0.6871f, 0.6980f, 0.6924f, 0.6770f, 0.6554f, 0.6294f, 0.6002f, 0.5686f, 0.5353f, 0.5007f, 0.4655f, 0.4301f, 0.3951f, 0.3612f, 0.3286f, 0.2974f, 0.2677f, 0.2392f, 0.2122f, 0.1867f, 0.1625f, 0.1403f, 0.1199f, 0.1012f, 0.0840f, 0.0681f, 0.0534f, 0.0400f, 0.0277f, 0.0168f},
    {0.5685f, 0.6377f, 0.6830f, 0.6947f, 0.6902f, 0.6761f, 0.6560f, 0.6316f, 0.6041f, 0.5743f, 0.5427f, 0.5099f, 0.4763f, 0.4424f, 0.4086f, 0.3755f, 0.3436f, 0.3130f, 0.2836f, 0.2554f, 0.2284f, 0.2026f, 0.1782f, 0.1555f, 0.1346f, 0.1152f, 0.0971f, 0.0802f, 0.0644f, 0.0496f, 0.0359f, 0.0233f},
    {0.5676f, 0.6363f, 0.6822f, 0.6947f, 0.6910f, 0.6781f, 0.6591f, 0.6361f, 0.6100f, 0.5817f, 0.5516f, 0.5202f, 0.4880f, 0.4553f, 0.4225f, 0.3902f, 0.3588f, 0.3285f, 0.2994f, 0.2713f, 0.2443f, 0.2184f, 0.1936f, 0.1705f, 0.1491f, 0.1291f, 0.1104f, 0.0926f, 0.0758f, 0.0600f, 0.0450f, 0.0309f},
    {0.5694f, 0.6378f, 0.6842f, 0.6973f, 0.6944f, 0.6823f, 0.6644f, 0.6424f, 0.6175f, 0.5904f, 0.5616f, 0.5315f, 0.5004f, 0.4687f, 0.4368f, 0.4051f, 0.3740f, 0.3440f, 0.3150f, 0.2870f, 0.2600f, 0.2339f, 0.2089f, 0.1854f, 0.1636f, 0.1431f, 0.1237f, 0.1052f, 0.0876f, 0.0707f, 0.0546f, 0.0393f},
    {0.5734f, 0.6415f, 0.6884f, 0.7021f, 0.6998f, 0.6884f, 0.6713f, 0.6502f, 0.6263f, 0.6002f, 0.5724f, 0.5434f, 0.5133f, 0.4824f, 0.4513f, 0.4201f, 0.3894f, 0.3595f, 0.3306f, 0.3026f, 0.2754f, 0.2492f, 0.2239f, 0.2001f, 0.1778f, 0.1569f, 0.1370f, 0.1179f, 0.0995f, 0.0818f, 0.0648f, 0.0483f},
    {0.5793f, 0.6470f, 0.6943f, 0.7085f, 0.7067f, 0.6959f, 0.6795f, 0.6591f, 0.6360f, 0.6108f, 0.5839f, 0.5557f, 0.5265f, 0.4964f, 0.4659f, 0.4352f, 0.4047f, 0.3749f, 0.3460f, 0.3179f, 0.2906f, 0.2642f, 0.2386f, 0.2144f, 0.1919f, 0.1705f, 0.1501f, 0.1305f, 0.1115f, 0.0931f, 0.0753f, 0.0579f},
    {0.5868f, 0.6539f, 0.7016f, 0.7162f, 0.7148f, 0.7044f, 0.6885f, 0.6688f, 0.6464f, 0.6219f, 0.5958f, 0.5683f, 0.5398f, 0.5104f, 0.4804f, 0.4501f, 0.4199f, 0.3901f, 0.3611f, 0.3329f, 0.3055f, 0.2788f, 0.2529f, 0.2285f, 0.2056f, 0.1838f, 0.1630f, 0.1430f, 0.1235f, 0.1045f, 0.0860f, 0.0679f},
    {0.5953f, 0.6618f, 0.7098f, 0.7246f, 0.7235f, 0.7136f, 0.6981f, 0.6789f, 0.6570f, 0.6331f, 0.6077f, 0.5808f, 0.5529f, 0.5241f, 0.4946f, 0.4647f, 0.4347f, 0.4049f, 0.3758f, 0.3475f, 0.3199f, 0.2930f, 0.2668f, 0.2421f, 0.2189f, 0.1968f, 0.1757f, 0.1552f, 0.1352f, 0.1158f, 0.0967f, 0.0780f},
    {0.6045f, 0.6703f, 0.7185f, 0.7335f, 0.7326f, 0.7230f, 0.7078f, 0.6890f, 0.6676f, 0.6443f, 0.6193f, 0.5931f, 0.5657f, 0.5374f, 0.5083f, 0.4787f, 0.4489f, 0.4192f, 0.3900f, 0.3615f, 0.3337f, 0.3066f, 0.2802f, 0.2552f, 0.2317f, 0.2093f, 0.1879f, 0.1670f, 0.1467f, 0.1269f, 0.1074f, 0.0882f},
    {0.6140f, 0.6790f, 0.7272f, 0.7423f, 0.7416f, 0.7322f, 0.7173f, 0.6989f, 0.6779f, 0.6549f, 0.6304f, 0.6047f, 0.5778f, 0.5500f, 0.5213f, 0.4920f, 0.4624f, 0.4328f, 0.4035f, 0.3748f, 0.3468f, 0.3195f, 0.2929f, 0.2676f, 0.2439f, 0.2213f, 0.1995f, 0.1785f, 0.1579f, 0.1377f, 0.1178f, 0.0984f},
    {0.6234f, 0.6874f, 0.7356f, 0.7507f, 0.7501f, 0.7408f, 0.7262f, 0.7080f, 0.6874f, 0.6648f, 0.6408f, 0.6154f, 0.5890f, 0.5616f, 0.5333f, 0.5044f, 0.4750f, 0.4454f, 0.4160f, 0.3872f, 0.3591f, 0.3316f, 0.3047f, 0.2793f, 0.2553f, 0.2326f, 0.2106f, 0.1893f, 0.1685f, 0.1481f, 0.1280f, 0.1083f},
    {0.6322f, 0.6951f, 0.7431f, 0.7583f, 0.7577f, 0.7485f, 0.7341f, 0.7162f, 0.6958f, 0.6736f, 0.6500f, 0.6250f, 0.5990f, 0.5720f, 0.5441f, 0.5155f, 0.4863f, 0.4569f, 0.4275f, 0.3986f, 0.3703f, 0.3427f, 0.3157f, 0.2901f, 0.2660f, 0.2431f, 0.2210f, 0.1995f, 0.1786f, 0.1580f, 0.1377f, 0.1179f},
    {0.6399f, 0.7017f, 0.7495f, 0.7646f, 0.7640f, 0.7550f, 0.7407f, 0.7230f, 0.7029f, 0.6810f, 0.6577f, 0.6332f, 0.6075f, 0.5809f, 0.5534f, 0.5251f, 0.4962f, 0.4670f, 0.4377f, 0.4087f, 0.3804f, 0.3527f, 0.3256f, 0.2999f, 0.2757f, 0.2527f, 0.2306f, 0.2090f, 0.1880f, 0.1673f, 0.1470f, 0.1271f},
    {0.6460f, 0.7067f, 0.7542f, 0.7692f, 0.7687f, 0.7597f, 0.7456f, 0.7281f, 0.7083f, 0.6867f, 0.6638f, 0.6396f, 0.6144f, 0.5881f, 0.5610f, 0.5331f, 0.5045f, 0.4755f, 0.4464f, 0.4175f, 0.3891f, 0.3614f, 0.3343f, 0.3086f, 0.2844f, 0.2614f, 0.2392f, 0.2177f, 0.1966f, 0.1759f, 0.1556f, 0.1357f},
    {0.6501f, 0.7097f, 0.7569f, 0.7717f, 0.7712f, 0.7623f, 0.7484f, 0.7312f, 0.7117f, 0.6904f, 0.6678f, 0.6440f, 0.6192f, 0.5934f, 0.5666f, 0.5391f, 0.5109f, 0.4822f, 0.4533f, 0.4246f, 0.3963f, 0.3687f, 0.3417f, 0.3160f, 0.2919f, 0.2690f, 0.2469f, 0.2255f, 0.2045f, 0.1839f, 0.1636f, 0.1438f},
    {0.6515f, 0.7102f, 0.7570f, 0.7718f, 0.7713f, 0.7626f, 0.7488f, 0.7319f, 0.7126f, 0.6917f, 0.6695f, 0.6461f, 0.6217f, 0.5964f, 0.5701f, 0.5430f, 0.5152f, 0.4869f, 0.4583f, 0.4299f, 0.4018f, 0.3744f, 0.3476f, 0.3221f, 0.2982f, 0.2754f, 0.2535f, 0.2322f, 0.2114f, 0.1909f, 0.1709f, 0.1512f},
    {0.6499f, 0.7078f, 0.7543f, 0.7690f, 0.7686f, 0.7600f, 0.7465f, 0.7298f, 0.7109f, 0.6904f, 0.6686f, 0.6457f, 0.6217f, 0.5968f, 0.5711f, 0.5444f, 0.5171f, 0.4893f, 0.4612f, 0.4331f, 0.4054f, 0.3783f, 0.3518f, 0.3266f, 0.3030f, 0.2805f, 0.2589f, 0.2379f, 0.2173f, 0.1971f, 0.1773f, 0.1579f},
    {0.6448f, 0.7021f, 0.7482f, 0.7629f, 0.7626f, 0.7542f, 0.7410f, 0.7247f, 0.7062f, 0.6861f, 0.6648f, 0.6423f, 0.6189f, 0.5945f, 0.5693f, 0.5433f, 0.5165f, 0.4892f, 0.4617f, 0.4341f, 0.4069f, 0.3803f, 0.3542f, 0.3295f, 0.3063f, 0.2842f, 0.2630f, 0.2423f, 0.2221f, 0.2023f, 0.1828f, 0.1638f}
};
inline float SheenAlbedo(float roughness, float NdotV) {
    const float2 p = saturate(float2(sqrt(NdotV), roughness)) * 31.0f;
    const uint2 i = min(uint2(p), uint2(30));
    const float2 f = p - float2(i);
    const float low = mix(SheenAlbedoTable[i.y][i.x], SheenAlbedoTable[i.y][i.x + 1], f.x);
    const float high = mix(SheenAlbedoTable[i.y + 1][i.x], SheenAlbedoTable[i.y + 1][i.x + 1], f.x);
    return mix(low, high, f.y);
}
// Samples the hemisphere about `n` uniformly.
inline float3 SheenSample(float3 n, float2 u) {
    const float radius = sqrt(saturate(1.0f - u.x * u.x)), phi = 2.0f * M_PI_F * u.y;
    return RayFrame(n) * float3(radius * cos(phi), radius * sin(phi), u.x);
}
inline float SheenPdf(float3 n, float3 l) { return dot(n, l) > 0.0f ? 0.5f / M_PI_F : 0.0f; }

// Specular transmission: the GGX distribution of the light direction mirrored through the surface, without a Fresnel factor.
inline float TransmissionValue(float3 n, float3 v, float3 l, float alpha, float ior) {
    const float alpha_t = applyIorToRoughness(alpha, ior);
    const float3 l_mirror = normalize(l + 2.0f * n * dot(-l, n));
    const float3 h = normalize(l_mirror + v);
    return D_GGX(saturate(dot(n, h)), alpha_t) * V_GGX(saturate(dot(n, l_mirror)), saturate(dot(n, v)), alpha_t);
}

// The metal and dielectric mix beneath the coat and sheen, with `weight` the part of the light that reaches it.
// Iridescence mixes in reflections with a thin-film Fresnel term that is constant over directions.
struct PbrBaseLayer {
    float3 SpecularF0, SpecularF90; // The GGX lobe of the metal and the dielectric, with their weights and multiple scattering
    float Diffuse; // The weight of the layer beneath the dielectric reflection
};
inline PbrBaseLayer BaseLayer(float2 terms, float weight, float3 base_color, float metallic, float3 dielectric_f0, float dielectric_f90, float iridescence, float3 iridescent_dielectric, float3 iridescent_metal) {
    const float metal = weight * metallic, dielectric = weight - metal;
    const float3 metal_f0 = metal * (1.0f - iridescence) * GgxCompensation(base_color, float3(1), terms) * base_color;
    const float3 metal_f90 = metal * (1.0f - iridescence) * GgxCompensation(base_color, float3(1), terms);
    const float3 metal_film = metal * iridescence * GgxCompensation(iridescent_metal, iridescent_metal, terms) * iridescent_metal;
    const float3 dielectric_scale = (1.0f - iridescence) * GgxCompensation(dielectric_f0, float3(dielectric_f90), terms);
    const float3 dielectric_film = iridescence * GgxCompensation(iridescent_dielectric, iridescent_dielectric, terms) * iridescent_dielectric;
    const float3 albedo = (dielectric_scale * dielectric_f0 + dielectric_film) * terms.x + (dielectric_scale * dielectric_f90 + dielectric_film) * terms.y;
    return {
        .SpecularF0 = metal_f0 + metal_film + dielectric * (dielectric_scale * dielectric_f0 + dielectric_film),
        .SpecularF90 = metal_f90 + metal_film + dielectric * (dielectric_scale * dielectric_f90 + dielectric_film),
        .Diffuse = dielectric * saturate(1.0f - max3(albedo.r, albedo.g, albedo.b)),
    };
}
