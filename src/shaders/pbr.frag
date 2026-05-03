#version 450

// Adapted from Khronos glTF-Sample-Renderer shader logic
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
#include "MaterialAlphaMode.glsl"
#include "PunctualLight.glsl"
#include "Transform.glsl"
#include "TRSUtils.glsl"
#include "brdf.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_LightBuffer, scalar) readonly buffer LightBufferBlock {
    PunctualLight Lights[];
} LightBuffers[];

layout(set = 0, binding = BINDING_ModelBuffer, scalar) readonly buffer ModelBufferBlock {
    Transform Models[];
} ModelBuffers[];

layout(set = 0, binding = BINDING_MaterialBuffer, scalar) readonly buffer MaterialBufferBlock {
    PBRMaterial Materials[];
} MaterialBuffers[];

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];
layout(set = 0, binding = BINDING_CubeSampler) uniform samplerCube CubeSamplers[];

#include "punctual.glsl"
#include "ibl.glsl"
#include "IridescenceBRDF.glsl"

// Spec constants: false disables the feature entirely (dead-code eliminated at specialization time).
// Defaults are all true so the unspecialized pipeline is the full-featured superset.
layout(constant_id = 0) const bool ENABLE_PUNCTUAL      = true;
layout(constant_id = 1) const bool ENABLE_TRANSMISSION  = true;
layout(constant_id = 2) const bool ENABLE_DIFFUSE_TRANS = true;
layout(constant_id = 3) const bool ENABLE_CLEARCOAT     = true;
layout(constant_id = 4) const bool ENABLE_SHEEN         = true;
layout(constant_id = 5) const bool ENABLE_ANISOTROPY    = true;
layout(constant_id = 6) const bool ENABLE_IRIDESCENCE   = true;

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

// KHR_materials_volume: Beer's law. attenuation_distance<=0 means infinite (no attenuation).
vec3 applyVolumeAttenuation(vec3 radiance, float dist, vec3 attenuation_color, float attenuation_distance) {
    if (attenuation_distance <= 0.0) return radiance;
    return pow(attenuation_color, vec3(dist / attenuation_distance)) * radiance;
}

// KHR_materials_transmission: direct-light BTDF.
// No Fresnel term — transmission D*V only (per glTF Sample Renderer).
vec3 getPunctualRadianceTransmission(vec3 n, vec3 v, vec3 l, float alpha_roughness, vec3 baseColor, float ior) {
    float tr = applyIorToRoughness(alpha_roughness, ior);
    vec3 l_mirror = normalize(l + 2.0 * n * dot(-l, n));
    vec3 h = normalize(l_mirror + v);
    return baseColor * D_GGX(clampedDot(n, h), tr) * V_GGX(clampedDot(n, l_mirror), clampedDot(n, v), tr);
}

// KHR_materials_volume: refraction ray inside the volume, scaled by world-space thickness.
// Caller subtracts this from point_to_light so the BTDF sees the light direction at the exit point.
vec3 getVolumeTransmissionRay(vec3 n, vec3 v, float world_thickness, float ior) {
    return normalize(refract(-v, normalize(n), 1.0 / ior)) * world_thickness;
}

// KHR_texture_transform: rotate the tangent basis around n by the normal map's UV rotation
// so sampled X/Y components stay aligned with the rotated UV axes.
mat3 GetNormalMapTBN(vec3 t, vec3 b, vec3 n, float uv_rotation) {
    const float c = cos(uv_rotation);
    const float s = sin(uv_rotation);
    return mat3(c * t - s * b, s * t + c * b, n);
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
        info.n = normalize(GetNormalMapTBN(t, b, ng, material.NormalTexture.UvRotation) * info.ntex);
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
    if (material.AlphaMode == MaterialAlphaMode_Opaque) {
        base_color.a = 1.0;
    }

    if (material.Unlit != 0u) {
        if (material.AlphaMode == MaterialAlphaMode_Mask) {
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
    float perceptual_roughness = material.RoughnessFactor;
    if (material.MetallicRoughnessTexture.Slot != INVALID_SLOT) {
        const vec4 metallic_roughness = texture(Samplers[nonuniformEXT(material.MetallicRoughnessTexture.Slot)], GetUv(material.MetallicRoughnessTexture));
        perceptual_roughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    perceptual_roughness = clamp(perceptual_roughness, 0.0, 1.0);
    const float alpha_roughness = perceptual_roughness * perceptual_roughness;

    // KHR_materials_sheen
    vec3 sheen_color = vec3(0.0);
    float sheen_roughness = 0.0;
    if (ENABLE_SHEEN) {
        sheen_color = material.Sheen.ColorFactor;
        if (material.Sheen.ColorTexture.Slot != INVALID_SLOT) {
            sheen_color *= texture(Samplers[nonuniformEXT(material.Sheen.ColorTexture.Slot)], GetUv(material.Sheen.ColorTexture)).rgb;
        }
        sheen_roughness = clamp(material.Sheen.RoughnessFactor, 0.0, 1.0);
        if (material.Sheen.RoughnessTexture.Slot != INVALID_SLOT) {
            sheen_roughness *= texture(Samplers[nonuniformEXT(material.Sheen.RoughnessTexture.Slot)], GetUv(material.Sheen.RoughnessTexture)).a;
        }
    }
    const bool has_sheen = ENABLE_SHEEN && any(greaterThan(sheen_color, vec3(0.0)));

    // KHR_materials_specular: specular_weight modulates dielectric F0 and F90.
    // Defaults (specularFactor=1, specularColorFactor=vec3(1), no textures) reproduce the standard 0.04 F0.
    float specular_weight = material.Specular.Factor;
    if (material.Specular.Texture.Slot != INVALID_SLOT) {
        specular_weight *= texture(Samplers[nonuniformEXT(material.Specular.Texture.Slot)], GetUv(material.Specular.Texture)).a;
    }
    float f0_ior_t = (material.Ior - 1.0) / (material.Ior + 1.0);
    float f0_ior = f0_ior_t * f0_ior_t;
    vec3 f0_dielectric = vec3(f0_ior) * material.Specular.ColorFactor;
    if (material.Specular.ColorTexture.Slot != INVALID_SLOT) {
        f0_dielectric *= texture(Samplers[nonuniformEXT(material.Specular.ColorTexture.Slot)], GetUv(material.Specular.ColorTexture)).rgb;
    }
    f0_dielectric = min(f0_dielectric, vec3(1.0));
    const vec3 f90_dielectric = vec3(specular_weight);

    // KHR_materials_transmission / KHR_materials_dispersion
    float transmission_factor = 0.0;
    if (ENABLE_TRANSMISSION) {
        transmission_factor = material.Transmission.Factor;
        if (material.Transmission.Texture.Slot != INVALID_SLOT) {
            transmission_factor *= texture(Samplers[nonuniformEXT(material.Transmission.Texture.Slot)], GetUv(material.Transmission.Texture)).r;
        }
    }
    // KHR_materials_diffuse_transmission
    float diffuse_transmission_factor = 0.0;
    vec3 diffuse_transmission_color = vec3(0.0);
    if (ENABLE_DIFFUSE_TRANS) {
        diffuse_transmission_factor = material.DiffuseTransmission.Factor;
        if (material.DiffuseTransmission.Texture.Slot != INVALID_SLOT) {
            diffuse_transmission_factor *= texture(Samplers[nonuniformEXT(material.DiffuseTransmission.Texture.Slot)], GetUv(material.DiffuseTransmission.Texture)).a;
        }
        diffuse_transmission_color = material.DiffuseTransmission.ColorFactor;
        if (material.DiffuseTransmission.ColorTexture.Slot != INVALID_SLOT) {
            diffuse_transmission_color *= texture(Samplers[nonuniformEXT(material.DiffuseTransmission.ColorTexture.Slot)], GetUv(material.DiffuseTransmission.ColorTexture)).rgb;
        }
    }
    // KHR_materials_volume: ThicknessFactor is model-space; multiply by world scale for Beer's law.
    float world_thickness = material.Volume.ThicknessFactor * WorldScale;
    if (material.Volume.ThicknessTexture.Slot != INVALID_SLOT) {
        world_thickness *= texture(Samplers[nonuniformEXT(material.Volume.ThicknessTexture.Slot)], GetUv(material.Volume.ThicknessTexture)).g;
    }

    // KHR_materials_clearcoat
    float clearcoat_factor = 0.0;
    float cc_perceptual_roughness = 0.0;
    float cc_alpha_roughness = 0.0;
    vec3 n_cc = normal_info.ng;
    float NdotV_cc = 0.0;
    if (ENABLE_CLEARCOAT) {
        clearcoat_factor = material.Clearcoat.Factor;
        if (material.Clearcoat.Texture.Slot != INVALID_SLOT) {
            clearcoat_factor *= texture(Samplers[nonuniformEXT(material.Clearcoat.Texture.Slot)], GetUv(material.Clearcoat.Texture)).r;
        }
        cc_perceptual_roughness = clamp(material.Clearcoat.RoughnessFactor, 0.0, 1.0);
        if (material.Clearcoat.RoughnessTexture.Slot != INVALID_SLOT) {
            cc_perceptual_roughness *= texture(Samplers[nonuniformEXT(material.Clearcoat.RoughnessTexture.Slot)], GetUv(material.Clearcoat.RoughnessTexture)).g;
        }
        cc_perceptual_roughness = clamp(cc_perceptual_roughness, 0.0, 1.0);
        cc_alpha_roughness = cc_perceptual_roughness * cc_perceptual_roughness;
        // Clearcoat normal defaults to the geometric normal; optionally overridden by its own normal map.
        // Uses the same tangent basis (t, b from normal_info) as the base material.
        if (material.Clearcoat.NormalTexture.Slot != INVALID_SLOT) {
            vec3 cc_ntex = texture(Samplers[nonuniformEXT(material.Clearcoat.NormalTexture.Slot)], GetUv(material.Clearcoat.NormalTexture)).rgb * 2.0 - vec3(1.0);
            cc_ntex *= vec3(material.Clearcoat.NormalScale, material.Clearcoat.NormalScale, 1.0);
            n_cc = normalize(GetNormalMapTBN(normal_info.t, normal_info.b, normal_info.ng, material.Clearcoat.NormalTexture.UvRotation) * normalize(cc_ntex));
        }
        NdotV_cc = clampedDot(n_cc, v);
    }
    const bool has_clearcoat = ENABLE_CLEARCOAT && clearcoat_factor > 0.0;

    // KHR_materials_anisotropy
    float anisotropy_strength = 0.0;
    vec3 anisotropic_t = vec3(0.0);
    vec3 anisotropic_b = vec3(0.0);
    if (ENABLE_ANISOTROPY) {
        anisotropy_strength = material.Anisotropy.Strength;
        // Pre-rotate the tangent-space direction by the material rotation angle.
        vec2 anisotropy_dir = vec2(cos(material.Anisotropy.Rotation), sin(material.Anisotropy.Rotation));
        if (material.Anisotropy.Texture.Slot != INVALID_SLOT) {
            const vec3 anisotropySample = texture(Samplers[nonuniformEXT(material.Anisotropy.Texture.Slot)], GetUv(material.Anisotropy.Texture)).rgb;
            // Texture RG encodes direction in [0,1]; remap to [-1,1] then rotate by material angle.
            const vec2 texDir = anisotropySample.xy * 2.0 - vec2(1.0);
            const mat2 rotMatrix = mat2(anisotropy_dir.x, anisotropy_dir.y, -anisotropy_dir.y, anisotropy_dir.x);
            anisotropy_dir = normalize(rotMatrix * texDir);
            anisotropy_strength *= anisotropySample.z;
        }
        anisotropy_strength = clamp(anisotropy_strength, 0.0, 1.0);
        // World-space anisotropy axes. anisotropic_b uses the geometric normal (same as reference).
        anisotropic_t = normalize(mat3(normal_info.t, normal_info.b, normal_info.n) * vec3(anisotropy_dir, 0.0));
        anisotropic_b = cross(normal_info.ng, anisotropic_t);
    }
    const bool has_anisotropy = ENABLE_ANISOTROPY && anisotropy_strength > 0.0;

    // KHR_materials_iridescence
    float iridescence_factor = 0.0;
    vec3 iridescence_fresnel_dielectric = vec3(0.0);
    vec3 iridescence_fresnel_metallic = vec3(0.0);
    if (ENABLE_IRIDESCENCE) {
        iridescence_factor = material.Iridescence.Factor;
        if (material.Iridescence.Texture.Slot != INVALID_SLOT) {
            iridescence_factor *= texture(Samplers[nonuniformEXT(material.Iridescence.Texture.Slot)], GetUv(material.Iridescence.Texture)).r;
        }
        iridescence_factor = clamp(iridescence_factor, 0.0, 1.0);
        float iridescenceThickness = material.Iridescence.ThicknessMaximum;
        if (material.Iridescence.ThicknessTexture.Slot != INVALID_SLOT) {
            const float t = texture(Samplers[nonuniformEXT(material.Iridescence.ThicknessTexture.Slot)], GetUv(material.Iridescence.ThicknessTexture)).g;
            iridescenceThickness = mix(material.Iridescence.ThicknessMinimum, material.Iridescence.ThicknessMaximum, t);
        }
        // Iridescence Fresnel is precomputed once at NdotV (not per-light), consistent with reference.
        iridescence_fresnel_dielectric = evalIridescence(1.0, material.Iridescence.Ior, NdotV, iridescenceThickness, f0_dielectric);
        iridescence_fresnel_metallic   = evalIridescence(1.0, material.Iridescence.Ior, NdotV, iridescenceThickness, base_color.rgb);
        if (iridescenceThickness == 0.0) iridescence_factor = 0.0;
    }
    const bool has_iridescence = ENABLE_IRIDESCENCE && iridescence_factor > 0.0;

    // Hoisted: NdotV is constant across lights.
    const float sheen_lut_ndotv = has_sheen ? albedoSheenScalingLUT(NdotV, sheen_roughness) : 0.0;

    vec3 direct_color = vec3(0.0);
    if (ENABLE_PUNCTUAL && SceneViewUBO.UseSceneLightsRender != 0u) {
        for (uint i = 0u; i < SceneViewUBO.LightCount; ++i) {
            const PunctualLight light = LightBuffers[nonuniformEXT(SceneViewUBO.LightSlot)].Lights[i];

            vec3 L;
            vec3 point_to_light;
            const vec3 light_intensity = getLightIntensity(light, WorldPosition, L, point_to_light);
            const vec3 H = normalize(L + v);
            const float NdotL = clampedDot(n, L);
            const float NdotH = clampedDot(n, H);
            const float VdotH = clampedDot(v, H);

            vec3 dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(VdotH));
            const vec3 metal_fresnel = F_Schlick(base_color.rgb, vec3(1.0), abs(VdotH));

            vec3 l_diffuse = light_intensity * NdotL * BRDF_lambertian(base_color.rgb);
            if (diffuse_transmission_factor > 0.0) {
                l_diffuse *= (1.0 - diffuse_transmission_factor);
                if (dot(n, L) < 0.0) {
                    vec3 l_diffuse_btdf = light_intensity * clampedDot(-n, L) * BRDF_lambertian(diffuse_transmission_color);
                    const vec3 l_mirror = normalize(L + 2.0 * n * dot(-L, n));
                    dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(clampedDot(v, normalize(l_mirror + v))));
                    l_diffuse_btdf = applyVolumeAttenuation(l_diffuse_btdf, world_thickness, material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
                    l_diffuse += l_diffuse_btdf * diffuse_transmission_factor;
                }
            }
            if (transmission_factor > 0.0) {
                const vec3 transmission_ray = getVolumeTransmissionRay(n, v, world_thickness, material.Ior);
                const vec3 transmit_l = safeNormalize(point_to_light - transmission_ray, L);
                vec3 l_transmit = light_intensity * getPunctualRadianceTransmission(n, v, transmit_l, alpha_roughness, base_color.rgb, material.Ior);
                l_transmit = applyVolumeAttenuation(l_transmit, length(transmission_ray), material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
                l_diffuse = mix(l_diffuse, l_transmit, transmission_factor);
            }
            const vec3 l_specular = light_intensity * NdotL * (has_anisotropy
                ? BRDF_specularGGXAnisotropy(alpha_roughness, anisotropy_strength, n, v, L, H, anisotropic_t, anisotropic_b)
                : BRDF_specularGGX(alpha_roughness, NdotL, NdotV, NdotH));

            vec3 l_metal_brdf = metal_fresnel * l_specular;
            vec3 l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);
            if (has_iridescence) {
                l_metal_brdf = mix(l_metal_brdf, l_specular * iridescence_fresnel_metallic, iridescence_factor);
                l_dielectric_brdf = mix(l_dielectric_brdf, mix(l_diffuse, l_specular, iridescence_fresnel_dielectric), iridescence_factor);
            }
            vec3 l_color = mix(l_dielectric_brdf, l_metal_brdf, metallic);
            if (has_sheen) {
                const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
                const float l_albedo_sheen_scaling = min(
                    1.0 - max_sheen * sheen_lut_ndotv,
                    1.0 - max_sheen * albedoSheenScalingLUT(NdotL, sheen_roughness));
                l_color = light_intensity * NdotL * BRDF_specularSheen(sheen_color, sheen_roughness, NdotL, NdotV, NdotH)
                    + l_color * l_albedo_sheen_scaling;
            }
            if (has_clearcoat) {
                const float NdotL_cc = clampedDot(n_cc, L);
                const float NdotH_cc = clampedDot(n_cc, H);
                // Intentionally diverges from glTF-Sample-Viewer: Fresnel at the microfacet
                // half-angle (VdotH) rather than a constant view-angle approximation (NdotV_cc).
                const vec3 F_cc = F_Schlick(vec3(f0_ior), vec3(1.0), abs(VdotH));
                const vec3 l_clearcoat = light_intensity * NdotL_cc * F_cc * BRDF_specularGGX(cc_alpha_roughness, NdotL_cc, NdotV_cc, NdotH_cc);
                l_color = l_color * (1.0 - clearcoat_factor * F_cc) + clearcoat_factor * l_clearcoat;
            }
            direct_color += l_color;
        }
    }

    vec3 f_diffuse = getDiffuseLight(n) * base_color.rgb;
    if (diffuse_transmission_factor > 0.0) {
        vec3 f_diffuse_transmission = getDiffuseLight(-n) * diffuse_transmission_color;
        f_diffuse_transmission = applyVolumeAttenuation(f_diffuse_transmission, world_thickness, material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
        f_diffuse = mix(f_diffuse, f_diffuse_transmission, diffuse_transmission_factor);
    }
    if (transmission_factor > 0.0) {
        vec3 f_transmission = getIBLVolumeRefraction(n, v, perceptual_roughness, material.Ior, material.Dispersion) * base_color.rgb;
        f_transmission = applyVolumeAttenuation(f_transmission, world_thickness, material.Volume.AttenuationColor, material.Volume.AttenuationDistance);
        f_diffuse = mix(f_diffuse, f_transmission, transmission_factor);
    }
    const vec3 f_specular_dielectric = has_anisotropy
        ? getIBLRadianceAnisotropy(n, v, perceptual_roughness, anisotropy_strength, anisotropic_b)
        : getIBLRadianceGGX(n, v, perceptual_roughness);
    const vec3 f_specular_metal = f_specular_dielectric;

    const vec2 ibl_brdf_f_ab = texture(Samplers[nonuniformEXT(SceneViewUBO.Ibl.BrdfLutSamplerSlot)],
                                       clamp(vec2(NdotV, perceptual_roughness), vec2(0.0), vec2(1.0))).rg;
    const vec3 f_metal_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, base_color.rgb, 1.0);
    vec3 f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

    const vec3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, f0_dielectric, specular_weight);
    vec3 f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

    if (has_iridescence) {
        f_metal_brdf_ibl = mix(f_metal_brdf_ibl, f_specular_metal * iridescence_fresnel_metallic, iridescence_factor);
        f_dielectric_brdf_ibl = mix(f_dielectric_brdf_ibl, mix(f_diffuse, f_specular_dielectric, iridescence_fresnel_dielectric), iridescence_factor);
    }
    vec3 indirect_color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, metallic);
    if (has_sheen) {
        const vec3 f_sheen = getIBLRadianceCharlie(n, v, sheen_roughness, sheen_color);
        const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
        const float albedo_sheen_scaling = 1.0 - max_sheen * sheen_lut_ndotv;
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
        cc_fresnel_ibl = getIBLGGXFresnel(n_cc, v, cc_perceptual_roughness, vec3(f0_ior), 1.0);
        const vec3 f_clearcoat_ibl = cc_fresnel_ibl * getIBLRadianceGGX(n_cc, v, cc_perceptual_roughness);
        indirect_color = indirect_color * (1.0 - clearcoat_factor * cc_fresnel_ibl) + clearcoat_factor * f_clearcoat_ibl;
    }

    vec3 color = direct_color + indirect_color;
    vec3 emissive = material.EmissiveFactor;
    if (material.EmissiveTexture.Slot != INVALID_SLOT) {
        emissive *= texture(Samplers[nonuniformEXT(material.EmissiveTexture.Slot)], GetUv(material.EmissiveTexture)).rgb;
    }
    if (has_clearcoat) emissive *= (1.0 - clearcoat_factor * cc_fresnel_ibl);
    color += emissive;

    if (material.AlphaMode == MaterialAlphaMode_Mask) {
        if (base_color.a < material.AlphaCutoff) discard;
        base_color.a = 1.0;
    }

    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionMode_Edit && SceneViewUBO.EditElement == Element_Face;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ? mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) : selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    color = toneMapPBRNeutral(color);
    color = linearTosRGB(color);
    OutColor = vec4(color, base_color.a);
}
