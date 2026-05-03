// Adapted from KhronosGroup/glTF-Sample-Renderer (ibl.glsl)

#ifndef IBL_GLSL
#define IBL_GLSL

vec3 getDiffuseLight(vec3 n) {
    vec4 texture_sample = texture(
        CubeSamplers[nonuniformEXT(SceneViewUBO.Ibl.DiffuseEnvSamplerSlot)],
        SceneViewUBO.EnvRotation * n
    );
    texture_sample.rgb *= SceneViewUBO.EnvIntensity;
    return texture_sample.rgb;
}

vec4 getSpecularSample(vec3 reflection, float lod) {
    vec4 texture_sample = textureLod(
        CubeSamplers[nonuniformEXT(SceneViewUBO.Ibl.SpecularEnvSamplerSlot)],
        SceneViewUBO.EnvRotation * reflection,
        lod
    );
    texture_sample.rgb *= SceneViewUBO.EnvIntensity;
    return texture_sample;
}

vec3 getIBLGGXFresnel(vec2 f_ab, float NdotV, float roughness, vec3 F0, float specular_weight) {
    const vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    const vec3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);
    const vec3 FssEss = specular_weight * (k_S * f_ab.x + f_ab.y);

    const float Ems = 1.0 - (f_ab.x + f_ab.y);
    const vec3 F_avg = specular_weight * (F0 + (1.0 - F0) / 21.0);
    const vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    return FssEss + FmsEms;
}

vec3 getIBLGGXFresnel(vec3 n, vec3 v, float roughness, vec3 F0, float specular_weight) {
    const float NdotV = clamp(dot(n, v), 0.0, 1.0);
    const vec2 f_ab = texture(Samplers[nonuniformEXT(SceneViewUBO.Ibl.BrdfLutSamplerSlot)],
                              clamp(vec2(NdotV, roughness), vec2(0.0), vec2(1.0))).rg;
    return getIBLGGXFresnel(f_ab, NdotV, roughness, F0, specular_weight);
}

vec3 getIBLRadianceGGX(vec3 n, vec3 v, float roughness) {
    const uint mip_count = max(SceneViewUBO.Ibl.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    return getSpecularSample(normalize(reflect(-v, n)), lod).rgb;
}

vec4 getSheenSample(vec3 reflection, float lod) {
    vec4 s = textureLod(CubeSamplers[nonuniformEXT(SceneViewUBO.Ibl.SheenEnvSamplerSlot)],
                        SceneViewUBO.EnvRotation * reflection, lod);
    s.rgb *= SceneViewUBO.EnvIntensity;
    return s;
}

float albedoSheenScalingLUT(float NdotV, float sheen_roughness) {
    return texture(Samplers[nonuniformEXT(SceneViewUBO.Ibl.SheenELutSamplerSlot)],
                   clamp(vec2(NdotV, sheen_roughness), vec2(0.0), vec2(1.0))).r;
}

// KHR_materials_anisotropy: bends the reflection vector toward the anisotropic specular lobe
// direction, then samples the prefiltered env map at that direction.
// Adapted from KhronosGroup/glTF-Sample-Renderer (ibl.glsl).
vec3 getIBLRadianceAnisotropy(vec3 n, vec3 v, float roughness, float anisotropy, vec3 anisotropy_dir) {
    const vec3 anisotropic_tangent = cross(anisotropy_dir, v);
    const vec3 anisotropic_normal  = cross(anisotropic_tangent, anisotropy_dir);
    const float bend_factor = 1.0 - anisotropy * (1.0 - roughness);
    const vec3 bent_normal = normalize(mix(anisotropic_normal, n, bend_factor * bend_factor * bend_factor * bend_factor));

    const uint mip_count = max(SceneViewUBO.Ibl.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    const vec3 reflection = normalize(reflect(-v, bent_normal));
    return getSpecularSample(reflection, lod).rgb;
}

// KHR_materials_ior / KHR_materials_transmission helpers.
float applyIorToRoughness(float roughness, float ior) {
    // IOR=1 → no microfacet roughening; IOR=1.5 → full roughness; per glTF Sample Renderer.
    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

// Sample the transmission framebuffer (mip chain of the pre-rendered scene without transmission objects)
// at the projected refracted exit point. LOD scales with applyIorToRoughness, matching glTF Sample Renderer.
vec3 sampleTransmissionFramebuffer(vec3 refracted_dir, vec3 world_pos, float world_thickness, float perceptual_roughness, float ior) {
    vec4 clip = SceneViewUBO.ViewProj * vec4(world_pos + refracted_dir * world_thickness, 1.0);
    vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    float lod = log2(float(max(SceneViewUBO.ViewportSize.x, 1.0))) * applyIorToRoughness(perceptual_roughness, ior);
    lod = clamp(lod, 0.0, float(max(SceneViewUBO.TransmissionFramebufferMipCount, 1u) - 1u));
    return textureLod(Samplers[nonuniformEXT(SceneViewUBO.TransmissionFramebufferSamplerSlot)], uv, lod).rgb;
}

// Sample the prefiltered specular env at the refracted direction (IBL approximation).
vec3 sampleIblRefraction(vec3 refracted_dir, float perceptual_roughness, float ior) {
    float lod = applyIorToRoughness(perceptual_roughness, ior) * float(max(SceneViewUBO.Ibl.SpecularEnvMipCount, 1u) - 1u);
    return getSpecularSample(refracted_dir, lod).rgb;
}

// Sample environment at the refracted ray direction. When `real`, samples the pre-rendered transmission
// framebuffer at the projected exit point; otherwise samples the prefiltered IBL.
// KHR_materials_dispersion: for dispersion>0, split IOR across RGB channels and sample each per-channel.
vec3 getVolumeRefraction(vec3 n, vec3 v, vec3 world_pos, float world_thickness, float perceptual_roughness, float ior, float dispersion, bool real) {
    #define SAMPLE_DIR(dir, channel_ior) (real \
        ? sampleTransmissionFramebuffer(dir, world_pos, world_thickness, perceptual_roughness, channel_ior) \
        : sampleIblRefraction(dir, perceptual_roughness, channel_ior))
    if (dispersion > 0.0) {
        float half_spread = (ior - 1.0) * 0.025 * dispersion;
        vec3 iors = vec3(ior - half_spread, ior, ior + half_spread);
        vec3 transmitted_light = vec3(0.0);
        for (int i = 0; i < 3; ++i) {
            vec3 refracted = normalize(refract(-v, n, 1.0 / iors[i]));
            transmitted_light[i] = SAMPLE_DIR(refracted, iors[i])[i];
        }
        return transmitted_light;
    }
    vec3 refracted = normalize(refract(-v, n, 1.0 / ior));
    return SAMPLE_DIR(refracted, ior);
    #undef SAMPLE_DIR
}

vec3 getIBLRadianceCharlie(vec3 n, vec3 v, float sheen_roughness, vec3 sheen_color) {
    const float NdotV = clamp(dot(n, v), 0.0, 1.0);
    const float lod = sheen_roughness * float(max(SceneViewUBO.Ibl.SheenEnvMipCount, 1u) - 1u);
    const vec3 reflection = normalize(reflect(-v, n));
    const float brdf = texture(Samplers[nonuniformEXT(SceneViewUBO.Ibl.CharlieLutSamplerSlot)],
                               clamp(vec2(NdotV, sheen_roughness), vec2(0.0), vec2(1.0))).b;
    return getSheenSample(reflection, lod).rgb * sheen_color * brdf;
}

#endif
