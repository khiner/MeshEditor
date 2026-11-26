#pragma once
// Adapted from KhronosGroup/glTF-Sample-Renderer (ibl.glsl)
#include "Bindless.metal"
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

template<typename SetT>
inline float3 getIBLRadianceGGX(const thread SceneT<SetT> &scene, float3 n, float3 v, float roughness) {
    const uint mip_count = max(scene.View.Ibl.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    return getSpecularSample(scene, normalize(reflect(-v, n)), lod).rgb;
}

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

template<typename SetT>
inline float3 sampleTransmissionFramebuffer(const thread SceneT<SetT> &scene, float3 refracted_dir, float3 world_pos, float world_thickness, float perceptual_roughness, float ior) {
    const float4 clip = scene.ViewProj() * float4(world_pos + refracted_dir * world_thickness, 1.0f);
    const float2 uv = ndc_to_uv(clip.xy / clip.w);
    float lod = log2(float(max(scene.View.ViewportSize[0], 1.0f))) * applyIorToRoughness(perceptual_roughness, ior);
    lod = clamp(lod, 0.0f, float(max(scene.View.TransmissionFramebufferMipCount, 1u) - 1u));
    return scene.SampleTexLod(scene.View.TransmissionFramebufferSamplerSlot, uv, lod).rgb;
}

template<typename SetT>
inline float3 sampleIblRefraction(const thread SceneT<SetT> &scene, float3 refracted_dir, float perceptual_roughness, float ior) {
    const float lod = applyIorToRoughness(perceptual_roughness, ior) * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u);
    return getSpecularSample(scene, refracted_dir, lod).rgb;
}

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

