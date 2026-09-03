// Adapted from KhronosGroup/glTF-Sample-Renderer (ibl.glsl)

#ifndef IBL_MSL
#define IBL_MSL

#include "Bindless.metal"
#include "brdf.metal"
#include "ScreenSpace.metal"

template<typename SetT>
inline float3 getDiffuseLight(const thread SceneT<SetT> &scene, float3 n) {
    float4 texture_sample = scene.SampleCube(scene.View.Ibl.DiffuseEnvSamplerSlot, scene.View.EnvRotation.Unpack() * n);
    return texture_sample.rgb * scene.View.EnvIntensity;
}

template<typename SetT>
inline float4 getSpecularSample(const thread SceneT<SetT> &scene, float3 reflection, float lod) {
    float4 texture_sample = scene.SampleCubeLod(scene.View.Ibl.SpecularEnvSamplerSlot, scene.View.EnvRotation.Unpack() * reflection, lod);
    texture_sample.rgb *= scene.View.EnvIntensity;
    return texture_sample;
}

inline float3 getIBLGGXFresnel(float2 f_ab, float NdotV, float roughness, float3 F0, float specular_weight) {
    const float3 Fr = max(float3(1.0f - roughness), F0) - F0;
    const float3 k_S = F0 + Fr * pow(1.0f - NdotV, 5.0f);
    const float3 FssEss = specular_weight * (k_S * f_ab.x + f_ab.y);

    const float Ems = 1.0f - (f_ab.x + f_ab.y);
    const float3 F_avg = specular_weight * (F0 + (1.0f - F0) / 21.0f);
    const float3 FmsEms = Ems * FssEss * F_avg / (1.0f - F_avg * Ems);
    return FssEss + FmsEms;
}

template<typename SetT>
inline float3 getIBLGGXFresnel(const thread SceneT<SetT> &scene, float3 n, float3 v, float roughness, float3 F0, float specular_weight) {
    const float NdotV = clamp(dot(n, v), 0.0f, 1.0f);
    const float2 f_ab = scene.SampleTex(scene.View.Ibl.BrdfLutSamplerSlot, clamp(float2(NdotV, roughness), float2(0.0f), float2(1.0f))).rg;
    return getIBLGGXFresnel(f_ab, NdotV, roughness, F0, specular_weight);
}

template<typename SetT>
inline float3 getIBLRadianceGGX(const thread SceneT<SetT> &scene, float3 n, float3 v, float roughness) {
    const uint mip_count = max(scene.View.Ibl.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    return getSpecularSample(scene, normalize(reflect(-v, n)), lod).rgb;
}

template<typename SetT>
inline float4 getSheenSample(const thread SceneT<SetT> &scene, float3 reflection, float lod) {
    float4 s = scene.SampleCubeLod(scene.View.Ibl.SheenEnvSamplerSlot, scene.View.EnvRotation.Unpack() * reflection, lod);
    s.rgb *= scene.View.EnvIntensity;
    return s;
}

template<typename SetT>
inline float albedoSheenScalingLUT(const thread SceneT<SetT> &scene, float NdotV, float sheen_roughness) {
    return scene.SampleTex(scene.View.Ibl.SheenELutSamplerSlot, clamp(float2(NdotV, sheen_roughness), float2(0.0f), float2(1.0f))).r;
}

// KHR_materials_anisotropy bends the reflection vector toward the anisotropic lobe before environment sampling.
template<typename SetT>
inline float3 getIBLRadianceAnisotropy(const thread SceneT<SetT> &scene, float3 n, float3 v, float roughness, float anisotropy, float3 anisotropy_dir) {
    const float3 anisotropic_tangent = cross(anisotropy_dir, v);
    const float3 anisotropic_normal = cross(anisotropic_tangent, anisotropy_dir);
    const float bend_factor = 1.0f - anisotropy * (1.0f - roughness);
    const float3 bent_normal = normalize(mix(anisotropic_normal, n, bend_factor * bend_factor * bend_factor * bend_factor));

    const uint mip_count = max(scene.View.Ibl.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    const float3 reflection = normalize(reflect(-v, bent_normal));
    return getSpecularSample(scene, reflection, lod).rgb;
}

inline float applyIorToRoughness(float roughness, float ior) {
    // IOR=1 gives no microfacet roughening, IOR=1.5 gives full roughness, per the glTF Sample Renderer.
    return roughness * clamp(ior * 2.0f - 2.0f, 0.0f, 1.0f);
}

// Samples the opaque-scene mip chain at the projected refracted exit point.
template<typename SetT>
inline float3 sampleTransmissionFramebuffer(const thread SceneT<SetT> &scene, float3 refracted_dir, float3 world_pos, float world_thickness, float perceptual_roughness, float ior) {
    const float4 clip = scene.ViewProj() * float4(world_pos + refracted_dir * world_thickness, 1.0f);
    const float2 uv = ndc_to_uv(clip.xy / clip.w);
    float lod = log2(float(max(scene.View.ViewportSize[0], 1.0f))) * applyIorToRoughness(perceptual_roughness, ior);
    lod = clamp(lod, 0.0f, float(max(scene.View.TransmissionFramebufferMipCount, 1u) - 1u));
    return scene.SampleTexLod(scene.View.TransmissionFramebufferSamplerSlot, uv, lod).rgb;
}

// Samples the prefiltered specular environment along the refracted direction.
template<typename SetT>
inline float3 sampleIblRefraction(const thread SceneT<SetT> &scene, float3 refracted_dir, float perceptual_roughness, float ior) {
    const float lod = applyIorToRoughness(perceptual_roughness, ior) * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u);
    return getSpecularSample(scene, refracted_dir, lod).rgb;
}

// Samples either the opaque-scene framebuffer or prefiltered environment along the refracted direction.
// Positive KHR_materials_dispersion splits the IOR into per-channel samples.
template<typename SetT>
inline float3 getVolumeRefraction(const thread SceneT<SetT> &scene, float3 n, float3 v, float3 world_pos, float world_thickness, float perceptual_roughness, float ior, float dispersion, bool real) {
    if (dispersion > 0.0f) {
        const float half_spread = (ior - 1.0f) * 0.025f * dispersion;
        const float3 iors = float3(ior - half_spread, ior, ior + half_spread);
        float3 transmitted_light = float3(0.0f);
        for (int i = 0; i < 3; ++i) {
            const float3 refracted = normalize(refract(-v, n, 1.0f / iors[i]));
            const float3 sampled = real ?
                sampleTransmissionFramebuffer(scene, refracted, world_pos, world_thickness, perceptual_roughness, iors[i]) :
                sampleIblRefraction(scene, refracted, perceptual_roughness, iors[i]);
            transmitted_light[i] = sampled[i];
        }
        return transmitted_light;
    }
    const float3 refracted = normalize(refract(-v, n, 1.0f / ior));
    return real ?
        sampleTransmissionFramebuffer(scene, refracted, world_pos, world_thickness, perceptual_roughness, ior) :
        sampleIblRefraction(scene, refracted, perceptual_roughness, ior);
}

template<typename SetT>
inline float3 getIBLRadianceCharlie(const thread SceneT<SetT> &scene, float3 n, float3 v, float sheen_roughness, float3 sheen_color) {
    const float NdotV = clamp(dot(n, v), 0.0f, 1.0f);
    const float lod = sheen_roughness * float(max(scene.View.Ibl.SheenEnvMipCount, 1u) - 1u);
    const float3 reflection = normalize(reflect(-v, n));
    const float brdf = scene.SampleTex(scene.View.Ibl.CharlieLutSamplerSlot, clamp(float2(NdotV, sheen_roughness), float2(0.0f), float2(1.0f))).b;
    return getSheenSample(scene, reflection, lod).rgb * sheen_color * brdf;
}

#endif
