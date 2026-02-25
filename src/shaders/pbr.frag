#version 450

// Adapted from Khronos glTF-Sample-Renderer shader logic, pulled 2026-02-16.
//  - material graph subset: core glTF metallic-roughness (baseColor, MR, normal, occlusion, emissive)
//  - KHR_materials_unlit: baseColor-only path, no lighting or IBL
//  - loop over bindless LightBuffer (LightCount/LightSlot)
//  - IBL block: diffuse env + specular prefiltered env + GGX BRDF LUT
//  - MeshEditor face-overlay behavior

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#include "SceneUBO.glsl"
#include "BindlessBindings.glsl"
#include "PBRMaterial.glsl"
#include "PunctualLight.glsl"
#include "WorldTransform.glsl"
#include "TRSUtils.glsl"
#include "brdf.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_LightBuffer, scalar) readonly buffer LightBufferBlock {
    PunctualLight Lights[];
} LightBuffers[];

layout(set = 0, binding = BINDING_ModelBuffer, scalar) readonly buffer ModelBufferBlock {
    WorldTransform Models[];
} ModelBuffers[];

layout(set = 0, binding = BINDING_MaterialBuffer, scalar) readonly buffer MaterialBufferBlock {
    PBRMaterial Materials[];
} MaterialBuffers[];

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];
layout(set = 0, binding = BINDING_CubeSampler) uniform samplerCube CubeSamplers[];

#include "punctual.glsl"
#include "ibl.glsl"
#include "IridescenceBRDF.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 3) flat in uint FaceOverlayFlags;
layout(location = 4) in vec2 TexCoord0;
layout(location = 5) in vec2 TexCoord1;
layout(location = 6) in vec2 TexCoord2;
layout(location = 7) in vec2 TexCoord3;
layout(location = 8) flat in uint MaterialIndex;
layout(location = 9) in vec4 VertexColor;
layout(location = 10) in vec4 WorldTangent;
layout(location = 11) flat in float WorldScale;

layout(location = 0) out vec4 OutColor;

const uint INVALID_SLOT = 0xffffffffu;
const uint ALPHA_OPAQUE = 0u;
const uint ALPHA_MASK = 1u;

struct NormalInfo {
    vec3 ng;
    vec3 t;
    vec3 b;
    vec3 n;
    vec3 ntex;
};

vec2 GetUv(uint uv_set) {
    if (uv_set == 0u) return TexCoord0;
    if (uv_set == 1u) return TexCoord1;
    if (uv_set == 2u) return TexCoord2;
    if (uv_set == 3u) return TexCoord3;
    return TexCoord0;
}

vec2 ApplyUvTransform(vec2 uv, vec2 uv_offset, vec2 uv_scale, float uv_rotation) {
    const float s = sin(uv_rotation);
    const float c = cos(uv_rotation);
    const mat3 rotation = mat3(
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0
    );
    const mat3 scale = mat3(
        uv_scale.x, 0.0, 0.0,
        0.0, uv_scale.y, 0.0,
        0.0, 0.0, 1.0
    );
    const mat3 translation = mat3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        uv_offset.x, uv_offset.y, 1.0
    );
    const mat3 uv_transform = translation * rotation * scale;
    return (uv_transform * vec3(uv, 1.0)).xy;
}

vec2 GetUv(uint uv_set, vec2 uv_offset, vec2 uv_scale, float uv_rotation) {
    return ApplyUvTransform(GetUv(uv_set), uv_offset, uv_scale, uv_rotation);
}

vec2 GetUv(TextureInfo tex) {
    return ApplyUvTransform(GetUv(tex.TexCoord), tex.UvOffset, tex.UvScale, tex.UvRotation);
}

float clampedDot(vec3 x, vec3 y) {
    return clamp(dot(x, y), 0.0, 1.0);
}

// KHR_materials_volume: Beer's law. attenuationDistance<=0 means infinite (no attenuation).
vec3 applyVolumeAttenuation(vec3 radiance, float dist, vec3 attenuationColor, float attenuationDistance) {
    if (attenuationDistance <= 0.0) return radiance;
    return pow(attenuationColor, vec3(dist / attenuationDistance)) * radiance;
}

// KHR_materials_transmission: direct-light BTDF.
// No Fresnel term — transmission D*V only (per glTF Sample Renderer).
vec3 getPunctualRadianceTransmission(vec3 n, vec3 v, vec3 l, float alphaRoughness, vec3 baseColor, float ior) {
    float tr = applyIorToRoughness(alphaRoughness, ior);
    vec3 l_mirror = normalize(l + 2.0 * n * dot(-l, n));
    vec3 h = normalize(l_mirror + v);
    return baseColor * D_GGX(clampedDot(n, h), tr) * V_GGX(clampedDot(n, l_mirror), clampedDot(n, v), tr);
}

NormalInfo GetNormalInfo(const PBRMaterial material) {
    const vec2 uv = GetUv(material.NormalTexture);
    vec3 ng = normalize(WorldNormal);
    vec3 t;
    vec3 b;
    if (length(WorldTangent.xyz) > 1e-8) {
        t = normalize(WorldTangent.xyz - ng * dot(ng, WorldTangent.xyz));
        b = normalize(cross(ng, t) * WorldTangent.w);
    } else {
        vec2 uv_dx = dFdx(uv);
        vec2 uv_dy = dFdy(uv);
        if (length(uv_dx) <= 1e-2) uv_dx = vec2(1.0, 0.0);
        if (length(uv_dy) <= 1e-2) uv_dy = vec2(0.0, 1.0);

        const float det = uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t;
        const vec3 t_ = abs(det) > 1e-8 ?
            (uv_dy.t * dFdx(WorldPosition) - uv_dx.t * dFdy(WorldPosition)) / det :
            vec3(1.0, 0.0, 0.0);
        t = normalize(t_ - ng * dot(ng, t_));
        b = cross(ng, t);
    }

    if (!IsFrontFacing(ng, WorldPosition)) {
        t *= -1.0;
        b *= -1.0;
        ng *= -1.0;
    }

    NormalInfo info;
    info.ng = ng;
    info.t = t;
    info.b = b;
    if (material.NormalTexture.Slot != INVALID_SLOT) {
        info.ntex = texture(Samplers[nonuniformEXT(material.NormalTexture.Slot)], uv).rgb * 2.0 - vec3(1.0);
        info.ntex *= vec3(material.NormalScale, material.NormalScale, 1.0);
        info.ntex = normalize(info.ntex);
        info.n = normalize(mat3(t, b, ng) * info.ntex);
    } else {
        info.ntex = vec3(0.0, 0.0, 1.0);
        info.n = ng;
    }
    return info;
}

void main() {
    const PBRMaterial material = MaterialBuffers[nonuniformEXT(SceneViewUBO.MaterialSlot)].Materials[MaterialIndex];
    if (material.DoubleSided == 0u && !IsFrontFacing(WorldNormal, WorldPosition)) discard;

    vec4 base_color = material.BaseColorFactor;
    if (material.BaseColorTexture.Slot != INVALID_SLOT) {
        base_color *= texture(Samplers[nonuniformEXT(material.BaseColorTexture.Slot)], GetUv(material.BaseColorTexture));
    }
    base_color *= VertexColor;
    if (material.AlphaMode == ALPHA_OPAQUE) {
        base_color.a = 1.0;
    }

    if (material.Unlit != 0u) {
        if (material.AlphaMode == ALPHA_MASK) {
            if (base_color.a < material.AlphaCutoff) discard;
            base_color.a = 1.0;
        }
        OutColor = vec4(linearTosRGB(toneMapPBRNeutral(base_color.rgb)), base_color.a);
        return;
    }

    const vec3 v = normalize(SceneViewUBO.CameraPosition - WorldPosition);
    const NormalInfo normal_info = GetNormalInfo(material);
    const vec3 n = normal_info.n;
    const float NdotV = clampedDot(n, v);

    float metallic = material.MetallicFactor;
    float perceptualRoughness = material.RoughnessFactor;
    if (material.MetallicRoughnessTexture.Slot != INVALID_SLOT) {
        const vec4 metallic_roughness = texture(Samplers[nonuniformEXT(material.MetallicRoughnessTexture.Slot)], GetUv(material.MetallicRoughnessTexture));
        perceptualRoughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);
    const float alphaRoughness = perceptualRoughness * perceptualRoughness;

    // KHR_materials_sheen
    vec3 sheenColor = material.Sheen.ColorFactor;
    if (material.Sheen.ColorTexture.Slot != INVALID_SLOT) {
        sheenColor *= texture(Samplers[nonuniformEXT(material.Sheen.ColorTexture.Slot)], GetUv(material.Sheen.ColorTexture)).rgb;
    }
    float sheenRoughness = clamp(material.Sheen.RoughnessFactor, 0.0, 1.0);
    if (material.Sheen.RoughnessTexture.Slot != INVALID_SLOT) {
        sheenRoughness *= texture(Samplers[nonuniformEXT(material.Sheen.RoughnessTexture.Slot)], GetUv(material.Sheen.RoughnessTexture)).a;
    }
    const bool has_sheen = any(greaterThan(sheenColor, vec3(0.0)));

    // KHR_materials_specular: specularWeight modulates dielectric F0 and F90.
    // Defaults (specularFactor=1, specularColorFactor=vec3(1), no textures) reproduce the standard 0.04 F0.
    float specularWeight = material.Specular.Factor;
    if (material.Specular.Texture.Slot != INVALID_SLOT) {
        specularWeight *= texture(Samplers[nonuniformEXT(material.Specular.Texture.Slot)], GetUv(material.Specular.Texture)).a;
    }
    float f0_ior = pow((material.Ior - 1.0) / (material.Ior + 1.0), 2.0);
    vec3 f0_dielectric = vec3(f0_ior) * material.Specular.ColorFactor;
    if (material.Specular.ColorTexture.Slot != INVALID_SLOT) {
        f0_dielectric *= texture(Samplers[nonuniformEXT(material.Specular.ColorTexture.Slot)], GetUv(material.Specular.ColorTexture)).rgb;
    }
    f0_dielectric = min(f0_dielectric, vec3(1.0));
    const vec3 f90_dielectric = vec3(specularWeight);

    // KHR_materials_transmission
    float transmissionFactor = material.Transmission.Factor;
    if (material.Transmission.Texture.Slot != INVALID_SLOT) {
        transmissionFactor *= texture(Samplers[nonuniformEXT(material.Transmission.Texture.Slot)], GetUv(material.Transmission.Texture)).r;
    }
    // KHR_materials_volume: ThicknessFactor is model-space; multiply by world scale for Beer's law.
    float worldThickness = material.Volume.ThicknessFactor * WorldScale;
    if (material.Volume.ThicknessTexture.Slot != INVALID_SLOT) {
        worldThickness *= texture(Samplers[nonuniformEXT(material.Volume.ThicknessTexture.Slot)], GetUv(material.Volume.ThicknessTexture)).g;
    }

    // KHR_materials_clearcoat
    float clearcoatFactor = material.Clearcoat.Factor;
    if (material.Clearcoat.Texture.Slot != INVALID_SLOT) {
        clearcoatFactor *= texture(Samplers[nonuniformEXT(material.Clearcoat.Texture.Slot)], GetUv(material.Clearcoat.Texture)).r;
    }
    float ccPerceptualRoughness = clamp(material.Clearcoat.RoughnessFactor, 0.0, 1.0);
    if (material.Clearcoat.RoughnessTexture.Slot != INVALID_SLOT) {
        ccPerceptualRoughness *= texture(Samplers[nonuniformEXT(material.Clearcoat.RoughnessTexture.Slot)], GetUv(material.Clearcoat.RoughnessTexture)).g;
    }
    ccPerceptualRoughness = clamp(ccPerceptualRoughness, 0.0, 1.0);
    const float ccAlphaRoughness = ccPerceptualRoughness * ccPerceptualRoughness;

    // Clearcoat normal defaults to the geometric normal; optionally overridden by its own normal map.
    // Uses the same tangent basis (t, b from normal_info) as the base material.
    vec3 n_cc = normal_info.ng;
    if (material.Clearcoat.NormalTexture.Slot != INVALID_SLOT) {
        vec3 cc_ntex = texture(Samplers[nonuniformEXT(material.Clearcoat.NormalTexture.Slot)], GetUv(material.Clearcoat.NormalTexture)).rgb * 2.0 - vec3(1.0);
        cc_ntex *= vec3(material.Clearcoat.NormalScale, material.Clearcoat.NormalScale, 1.0);
        n_cc = normalize(mat3(normal_info.t, normal_info.b, normal_info.ng) * normalize(cc_ntex));
    }
    const float NdotV_cc = clampedDot(n_cc, v);
    const bool has_clearcoat = clearcoatFactor > 0.0;

    // KHR_materials_anisotropy
    float anisotropyStrength = material.Anisotropy.Strength;
    // Pre-rotate the tangent-space direction by the material rotation angle.
    vec2 anisotropyDirection = vec2(cos(material.Anisotropy.Rotation), sin(material.Anisotropy.Rotation));
    if (material.Anisotropy.Texture.Slot != INVALID_SLOT) {
        const vec3 anisotropySample = texture(Samplers[nonuniformEXT(material.Anisotropy.Texture.Slot)], GetUv(material.Anisotropy.Texture)).rgb;
        // Texture RG encodes direction in [0,1]; remap to [-1,1] then rotate by material angle.
        const vec2 texDir = anisotropySample.xy * 2.0 - vec2(1.0);
        const mat2 rotMatrix = mat2(anisotropyDirection.x, anisotropyDirection.y, -anisotropyDirection.y, anisotropyDirection.x);
        anisotropyDirection = normalize(rotMatrix * texDir);
        anisotropyStrength *= anisotropySample.z;
    }
    anisotropyStrength = clamp(anisotropyStrength, 0.0, 1.0);
    // World-space anisotropy axes. anisotropicB uses the geometric normal (same as reference).
    const vec3 anisotropicT = normalize(mat3(normal_info.t, normal_info.b, normal_info.n) * vec3(anisotropyDirection, 0.0));
    const vec3 anisotropicB = cross(normal_info.ng, anisotropicT);
    const bool has_anisotropy = anisotropyStrength > 0.0;

    // KHR_materials_iridescence
    float iridescenceFactor = material.Iridescence.Factor;
    if (material.Iridescence.Texture.Slot != INVALID_SLOT) {
        iridescenceFactor *= texture(Samplers[nonuniformEXT(material.Iridescence.Texture.Slot)], GetUv(material.Iridescence.Texture)).r;
    }
    iridescenceFactor = clamp(iridescenceFactor, 0.0, 1.0);
    float iridescenceThickness = material.Iridescence.ThicknessMaximum;
    if (material.Iridescence.ThicknessTexture.Slot != INVALID_SLOT) {
        const float t = texture(Samplers[nonuniformEXT(material.Iridescence.ThicknessTexture.Slot)], GetUv(material.Iridescence.ThicknessTexture)).g;
        iridescenceThickness = mix(material.Iridescence.ThicknessMinimum, material.Iridescence.ThicknessMaximum, t);
    }
    // Iridescence Fresnel is precomputed once at NdotV (not per-light), consistent with reference.
    const vec3 iridescenceFresnel_dielectric = evalIridescence(1.0, material.Iridescence.Ior, NdotV, iridescenceThickness, f0_dielectric);
    const vec3 iridescenceFresnel_metallic   = evalIridescence(1.0, material.Iridescence.Ior, NdotV, iridescenceThickness, base_color.rgb);
    if (iridescenceThickness == 0.0) iridescenceFactor = 0.0;
    const bool has_iridescence = iridescenceFactor > 0.0;

    vec3 direct_color = vec3(0.0);
    if (SceneViewUBO.UseSceneLightsRender != 0u) {
        for (uint i = 0u; i < SceneViewUBO.LightCount; ++i) {
            const PunctualLight light = LightBuffers[nonuniformEXT(SceneViewUBO.LightSlot)].Lights[i];

            vec3 L;
            const vec3 light_intensity = getLightIntensity(light, WorldPosition, L);
            const vec3 H = normalize(L + v);
            const float NdotL = clampedDot(n, L);
            const float NdotH = clampedDot(n, H);
            const float VdotH = clampedDot(v, H);

            const vec3 dielectric_fresnel = F_Schlick(f0_dielectric * specularWeight, f90_dielectric, abs(VdotH));
            const vec3 metal_fresnel = F_Schlick(base_color.rgb, vec3(1.0), abs(VdotH));

            vec3 l_diffuse = light_intensity * NdotL * BRDF_lambertian(base_color.rgb);
            if (transmissionFactor > 0.0) {
                vec3 l_transmit = light_intensity * getPunctualRadianceTransmission(n, v, L, alphaRoughness, base_color.rgb, material.Ior);
                l_transmit = applyVolumeAttenuation(l_transmit, worldThickness, material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
                l_diffuse = mix(l_diffuse, l_transmit, transmissionFactor);
            }
            const vec3 l_specular = light_intensity * NdotL * (has_anisotropy
                ? BRDF_specularGGXAnisotropy(alphaRoughness, anisotropyStrength, n, v, L, H, anisotropicT, anisotropicB)
                : BRDF_specularGGX(alphaRoughness, NdotL, NdotV, NdotH));

            vec3 l_metal_brdf = metal_fresnel * l_specular;
            vec3 l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);
            if (has_iridescence) {
                l_metal_brdf = mix(l_metal_brdf, l_specular * iridescenceFresnel_metallic, iridescenceFactor);
                l_dielectric_brdf = mix(l_dielectric_brdf, mix(l_diffuse, l_specular, iridescenceFresnel_dielectric), iridescenceFactor);
            }
            vec3 l_color = mix(l_dielectric_brdf, l_metal_brdf, metallic);
            if (has_sheen) {
                const float max_sheen = max(sheenColor.r, max(sheenColor.g, sheenColor.b));
                const float l_albedo_sheen_scaling = min(
                    1.0 - max_sheen * albedoSheenScalingLUT(NdotV, sheenRoughness),
                    1.0 - max_sheen * albedoSheenScalingLUT(NdotL, sheenRoughness));
                l_color = light_intensity * NdotL * BRDF_specularSheen(sheenColor, sheenRoughness, NdotL, NdotV, NdotH)
                    + l_color * l_albedo_sheen_scaling;
            }
            if (has_clearcoat) {
                const float NdotL_cc = clampedDot(n_cc, L);
                const float NdotH_cc = clampedDot(n_cc, H);
                // Intentionally diverges from glTF-Sample-Viewer: Fresnel at the microfacet
                // half-angle (VdotH) rather than a constant view-angle approximation (NdotV_cc).
                const vec3 F_cc = F_Schlick(vec3(f0_ior), vec3(1.0), abs(VdotH));
                const vec3 l_clearcoat = light_intensity * NdotL_cc * F_cc * BRDF_specularGGX(ccAlphaRoughness, NdotL_cc, NdotV_cc, NdotH_cc);
                l_color = l_color * (1.0 - clearcoatFactor * F_cc) + clearcoatFactor * l_clearcoat;
            }
            direct_color += l_color;
        }
    }

    vec3 f_diffuse = getDiffuseLight(n) * base_color.rgb;
    if (transmissionFactor > 0.0) {
        vec3 f_transmission = getIBLVolumeRefraction(n, v, perceptualRoughness, material.Ior) * base_color.rgb;
        f_transmission = applyVolumeAttenuation(f_transmission, worldThickness, material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
        f_diffuse = mix(f_diffuse, f_transmission, transmissionFactor);
    }
    const vec3 f_specular_dielectric = has_anisotropy
        ? getIBLRadianceAnisotropy(n, v, perceptualRoughness, anisotropyStrength, anisotropicB)
        : getIBLRadianceGGX(n, v, perceptualRoughness);
    const vec3 f_specular_metal = f_specular_dielectric;

    const vec3 f_metal_fresnel_ibl = getIBLGGXFresnel(n, v, perceptualRoughness, base_color.rgb, 1.0);
    vec3 f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

    const vec3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(n, v, perceptualRoughness, f0_dielectric, specularWeight);
    vec3 f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

    if (has_iridescence) {
        f_metal_brdf_ibl = mix(f_metal_brdf_ibl, f_specular_metal * iridescenceFresnel_metallic, iridescenceFactor);
        f_dielectric_brdf_ibl = mix(f_dielectric_brdf_ibl, mix(f_diffuse, f_specular_dielectric, iridescenceFresnel_dielectric), iridescenceFactor);
    }
    vec3 indirect_color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, metallic);
    if (has_sheen) {
        const vec3 f_sheen = getIBLRadianceCharlie(n, v, sheenRoughness, sheenColor);
        const float max_sheen = max(sheenColor.r, max(sheenColor.g, sheenColor.b));
        const float albedo_sheen_scaling = 1.0 - max_sheen * albedoSheenScalingLUT(NdotV, sheenRoughness);
        indirect_color = f_sheen + indirect_color * albedo_sheen_scaling;
    }
    if (material.OcclusionTexture.Slot != INVALID_SLOT) {
        const float ao = texture(Samplers[nonuniformEXT(material.OcclusionTexture.Slot)], GetUv(material.OcclusionTexture)).r;
        indirect_color *= (1.0 + material.OcclusionStrength * (ao - 1.0));
    }
    vec3 cc_fresnel_ibl = vec3(0.0);
    if (has_clearcoat) {
        // Intentionally diverges from glTF-Sample-Viewer: split-sum BRDF LUT Fresnel
        // (with multi-scattering energy compensation) rather than F_Schlick(F0, F90, NdotV_cc).
        cc_fresnel_ibl = getIBLGGXFresnel(n_cc, v, ccPerceptualRoughness, vec3(f0_ior), 1.0);
        const vec3 f_clearcoat_ibl = cc_fresnel_ibl * getIBLRadianceGGX(n_cc, v, ccPerceptualRoughness);
        indirect_color = indirect_color * (1.0 - clearcoatFactor * cc_fresnel_ibl) + clearcoatFactor * f_clearcoat_ibl;
    }

    vec3 color = direct_color + indirect_color;
    vec3 emissive = material.EmissiveFactor;
    if (material.EmissiveTexture.Slot != INVALID_SLOT) {
        emissive *= texture(Samplers[nonuniformEXT(material.EmissiveTexture.Slot)], GetUv(material.EmissiveTexture)).rgb;
    }
    if (has_clearcoat) emissive *= (1.0 - clearcoatFactor * cc_fresnel_ibl);
    color += emissive;

    if (material.AlphaMode == ALPHA_MASK) {
        if (base_color.a < material.AlphaCutoff) discard;
        base_color.a = 1.0;
    }

    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionModeEdit && SceneViewUBO.EditElement == EditElementFace;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ? mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) : selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    color = toneMapPBRNeutral(color);
    color = linearTosRGB(color);
    OutColor = vec4(color, base_color.a);
}
