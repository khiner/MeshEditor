#version 450

// Adapted from Khronos glTF-Sample-Renderer shader logic, pulled 2026-02-16.
//  - material graph subset: core glTF metallic-roughness (baseColor, MR, normal, occlusion, emissive)
//  - loop over bindless LightBuffer (LightCount/LightSlot)
//  - IBL block: diffuse env + specular prefiltered env + GGX BRDF LUT
//  - MeshEditor face-overlay behavior
//  - no glTF material extensions yet (lighting-first validation)

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

float clampedDot(vec3 x, vec3 y) {
    return clamp(dot(x, y), 0.0, 1.0);
}

NormalInfo GetNormalInfo(const PBRMaterial material) {
    const vec2 uv = GetUv(material.NormalTexCoord, material.NormalUvOffset, material.NormalUvScale, material.NormalUvRotation);
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

    if (!gl_FrontFacing) {
        t *= -1.0;
        b *= -1.0;
        ng *= -1.0;
    }

    NormalInfo info;
    info.ng = ng;
    info.t = t;
    info.b = b;
    if (material.NormalTexture != INVALID_SLOT) {
        info.ntex = texture(Samplers[nonuniformEXT(material.NormalTexture)], uv).rgb * 2.0 - vec3(1.0);
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
    if (material.DoubleSided == 0u && !gl_FrontFacing) discard;

    vec4 base_color = material.BaseColorFactor;
    if (material.BaseColorTexture != INVALID_SLOT) {
        const vec2 base_color_uv = GetUv(
            material.BaseColorTexCoord,
            material.BaseColorUvOffset,
            material.BaseColorUvScale,
            material.BaseColorUvRotation
        );
        base_color *= texture(Samplers[nonuniformEXT(material.BaseColorTexture)], base_color_uv);
    }
    base_color *= VertexColor;
    if (material.AlphaMode == ALPHA_OPAQUE) {
        base_color.a = 1.0;
    }

    const vec3 v = normalize(SceneViewUBO.CameraPosition - WorldPosition);
    const NormalInfo normal_info = GetNormalInfo(material);
    const vec3 n = normal_info.n;
    const float NdotV = clampedDot(n, v);

    float metallic = material.MetallicFactor;
    float perceptualRoughness = material.RoughnessFactor;
    if (material.MetallicRoughnessTexture != INVALID_SLOT) {
        const vec2 metallic_roughness_uv = GetUv(
            material.MetallicRoughnessTexCoord,
            material.MetallicRoughnessUvOffset,
            material.MetallicRoughnessUvScale,
            material.MetallicRoughnessUvRotation
        );
        const vec4 metallic_roughness = texture(Samplers[nonuniformEXT(material.MetallicRoughnessTexture)], metallic_roughness_uv);
        perceptualRoughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);
    const float alphaRoughness = perceptualRoughness * perceptualRoughness;

    const float specularWeight = 1.0;
    const vec3 f0_dielectric = vec3(0.04);
    const vec3 f90_dielectric = vec3(1.0);

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

            const vec3 l_diffuse = light_intensity * NdotL * BRDF_lambertian(base_color.rgb);
            const vec3 l_specular = light_intensity * NdotL * BRDF_specularGGX(alphaRoughness, NdotL, NdotV, NdotH);

            const vec3 l_metal_brdf = metal_fresnel * l_specular;
            const vec3 l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);
            const vec3 l_color = mix(l_dielectric_brdf, l_metal_brdf, metallic);
            direct_color += l_color;
        }
    }

    const vec3 f_diffuse = getDiffuseLight(n) * base_color.rgb;
    const vec3 f_specular_dielectric = getIBLRadianceGGX(n, v, perceptualRoughness);
    const vec3 f_specular_metal = f_specular_dielectric;

    const vec3 f_metal_fresnel_ibl = getIBLGGXFresnel(n, v, perceptualRoughness, base_color.rgb, 1.0);
    const vec3 f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

    const vec3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(n, v, perceptualRoughness, f0_dielectric, specularWeight);
    const vec3 f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

    vec3 indirect_color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, metallic);
    if (material.OcclusionTexture != INVALID_SLOT) {
        const vec2 occlusion_uv = GetUv(
            material.OcclusionTexCoord,
            material.OcclusionUvOffset,
            material.OcclusionUvScale,
            material.OcclusionUvRotation
        );
        const float ao = texture(Samplers[nonuniformEXT(material.OcclusionTexture)], occlusion_uv).r;
        indirect_color *= (1.0 + material.OcclusionStrength * (ao - 1.0));
    }

    vec3 color = direct_color + indirect_color;
    vec3 emissive = material.EmissiveFactor;
    if (material.EmissiveTexture != INVALID_SLOT) {
        const vec2 emissive_uv = GetUv(
            material.EmissiveTexCoord,
            material.EmissiveUvOffset,
            material.EmissiveUvScale,
            material.EmissiveUvRotation
        );
        emissive *= texture(Samplers[nonuniformEXT(material.EmissiveTexture)], emissive_uv).rgb;
    }
    color += emissive;

    if (material.AlphaMode == ALPHA_MASK) {
        if (base_color.a < material.AlphaCutoff) discard;
        base_color.a = 1.0;
    }

    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionModeEdit &&
            SceneViewUBO.EditElement == EditElementFace;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ?
            mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) :
            selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    color = toneMapPBRNeutral(color);
    color = linearTosRGB(color);
    OutColor = vec4(color, base_color.a);
}
