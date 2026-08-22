#ifndef PBR_MSL
#define PBR_MSL

// Adapted from Khronos glTF-Sample-Renderer shader logic
//  - material graph subset: core glTF metallic-roughness (baseColor, MR, normal, occlusion, emissive)
//  - KHR_materials_unlit: baseColor-only path, no lighting or IBL
//  - loop over the bindless light buffer (LightCount/LightSlot)
//  - IBL block: diffuse env + specular prefiltered env + GGX BRDF LUT
//  - MeshEditor face-overlay behavior
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "DebugChannel.metal"
#include "MaterialAlphaMode.metal"
#include "brdf.metal"
#include "tonemapping.metal"
#include "punctual.metal"
#include "ibl.metal"
#include "IridescenceBRDF.metal"
#include "PbrConstant.metal"
#include "Velocity.metal"
#include "VisibilityDecode.metal"

// Each Enable* constant disables its feature entirely when false, and the compiler removes it.
// Defaults are all true so the unspecialized pipeline is the full-featured superset.
// TransmissionPrepass renders into the transmission framebuffer: it discards transmission materials
// (they must not sample their own attachment) and skips exposure, which the main pass applies after
// sampling. VelocityOutput writes the screen motion the velocity pass reads.

constant uint INVALID_MATERIAL_SLOT = 0xffffffffu;

// The shaded color, plus the screen motion when the velocity variant is selected.
struct PbrTargets {
    float4 Color [[color(0)]];
    float4 Motion [[color(1)]] [[function_constant(VelocityOutput)]];
};

struct NormalInfo {
    float3 ng;
    float3 t;
    float3 b;
    float3 n;
    float3 ntex;
};

inline float clampedDot(float3 x, float3 y) { return clamp(dot(x, y), 0.0f, 1.0f); }

inline float2 ProjectToNdc(float4x4 view_proj, float3 world_pos) {
    const float4 clip = view_proj * float4(world_pos, 1.0f);
    return clip.xy / clip.w;
}

inline float2 ApplyUvTransform(float2 uv, float2 uv_offset, float2 uv_scale, float uv_rotation) {
    const float s = sin(uv_rotation);
    const float c = cos(uv_rotation);
    const float3x3 rotation = float3x3(float3(c, -s, 0.0f), float3(s, c, 0.0f), float3(0.0f, 0.0f, 1.0f));
    const float3x3 scale = float3x3(float3(uv_scale.x, 0.0f, 0.0f), float3(0.0f, uv_scale.y, 0.0f), float3(0.0f, 0.0f, 1.0f));
    const float3x3 translation = float3x3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(uv_offset.x, uv_offset.y, 1.0f));
    const float3x3 uv_transform = translation * rotation * scale;
    return (uv_transform * float3(uv, 1.0f)).xy;
}

// KHR_materials_volume: Beer's law. An attenuation distance of zero or less means no attenuation.
inline float3 applyVolumeAttenuation(float3 radiance, float dist, float3 attenuation_color, float attenuation_distance) {
    if (attenuation_distance <= 0.0f) return radiance;
    return pow(attenuation_color, float3(dist / attenuation_distance)) * radiance;
}

// KHR_materials_transmission: direct-light BTDF.
// No Fresnel term, transmission D*V only, per the glTF Sample Renderer.
inline float3 getPunctualRadianceTransmission(float3 n, float3 v, float3 l, float alpha_roughness, float3 baseColor, float ior) {
    const float tr = applyIorToRoughness(alpha_roughness, ior);
    const float3 l_mirror = normalize(l + 2.0f * n * dot(-l, n));
    const float3 h = normalize(l_mirror + v);
    return baseColor * D_GGX(clampedDot(n, h), tr) * V_GGX(clampedDot(n, l_mirror), clampedDot(n, v), tr);
}

// KHR_materials_volume: refraction ray inside the volume, scaled by world-space thickness.
// The caller subtracts this from point_to_light so the BTDF sees the light direction at the exit point.
inline float3 getVolumeTransmissionRay(float3 n, float3 v, float world_thickness, float ior) {
    return normalize(refract(-v, normalize(n), 1.0f / ior)) * world_thickness;
}

// KHR_texture_transform: rotate the tangent basis around n by the normal map's UV rotation
// so sampled X and Y components stay aligned with the rotated UV axes.
inline float3x3 GetNormalMapTBN(float3 t, float3 b, float3 n, float uv_rotation) {
    const float c = cos(uv_rotation);
    const float s = sin(uv_rotation);
    return float3x3(c * t - s * b, s * t + c * b, n);
}

// Everything the material graph reads: the bindless table, this fragment's varyings, and the
// helpers that resolve a material's texture coordinates.
struct PbrContext {
    Scene S;
    const thread MeshVaryings &In;
    float2 UvDx[4];
    float2 UvDy[4];
    bool ExplicitGradients;

    float2 GetUv(uint uv_set) const {
        if (uv_set == 1u) return In.TexCoord1;
        if (uv_set == 2u) return In.TexCoord2;
        if (uv_set == 3u) return In.TexCoord3;
        return In.TexCoord0;
    }
    float2 GetUv(TextureInfo tex) const {
        return ApplyUvTransform(GetUv(tex.TexCoord), float2(tex.UvOffset), float2(tex.UvScale), tex.UvRotation);
    }
    float2 TransformUvGradient(float2 gradient, TextureInfo tex) const {
        const float s = sin(tex.UvRotation);
        const float c = cos(tex.UvRotation);
        const float2 scaled = gradient * float2(tex.UvScale);
        return float2(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y);
    }
    float4 SampleTexture(TextureInfo tex) const {
        const float2 uv = GetUv(tex);
        if (!ExplicitGradients) return S.SampleTex(tex.Slot, uv);
        const uint set = min(tex.TexCoord, 3u);
        return S.SampleTexGrad(tex.Slot, uv, TransformUvGradient(UvDx[set], tex), TransformUvGradient(UvDy[set], tex));
    }

    NormalInfo GetNormalInfo(device const PBRMaterial &material) const {
        const float2 uv = GetUv(material.NormalTexture);
        float3 ng = normalize(ShadingWorldNormal(In));
        float3 t;
        float3 b;
        if (length(In.WorldTangent.xyz) > 1e-8f) {
            t = normalize(In.WorldTangent.xyz - ng * dot(ng, In.WorldTangent.xyz));
            b = normalize(cross(ng, t) * In.WorldTangent.w);
        } else {
            // A texture coordinate that barely moves down the screen falls back to a fixed
            // downward axis, so the frame stays defined where the mapping is degenerate.
            const float3 pos_dx = dfdx(In.WorldPosition);
            const float3 pos_dy = dfdy(In.WorldPosition);
            float2 uv_dx = dfdx(uv);
            float2 uv_dy = dfdy(uv);
            if (length(uv_dx) <= 1e-2f) uv_dx = float2(1.0f, 0.0f);
            if (length(uv_dy) <= 1e-2f) uv_dy = float2(0.0f, 1.0f);

            const float det = uv_dx.x * uv_dy.y - uv_dy.x * uv_dx.y;
            const float3 t_ = abs(det) > 1e-8f ?
                (uv_dy.y * pos_dx - uv_dx.y * pos_dy) / det :
                float3(1.0f, 0.0f, 0.0f);
            t = normalize(t_ - ng * dot(ng, t_));
            b = cross(ng, t);
        }

        if (!IsFrontFacing(S, ng, In.WorldPosition)) {
            t *= -1.0f;
            b *= -1.0f;
            ng *= -1.0f;
        }

        NormalInfo info;
        info.ng = ng;
        info.t = t;
        info.b = b;
        if (material.NormalTexture.Slot != INVALID_MATERIAL_SLOT) {
            info.ntex = SampleTexture(material.NormalTexture).rgb * 2.0f - float3(1.0f);
            info.ntex *= float3(material.NormalScale, material.NormalScale, 1.0f);
            info.ntex = normalize(info.ntex);
            info.n = normalize(GetNormalMapTBN(t, b, ng, material.NormalTexture.UvRotation) * info.ntex);
        } else {
            info.ntex = float3(0.0f, 0.0f, 1.0f);
            info.n = ng;
        }
        return info;
    }
};

inline PbrTargets ShadePbr(
    MeshVaryings in, uint topology, float2 point_coord, const thread Scene &scene,
    constant SceneViewUBO &view, const thread PbrContext &ctx
) {
    PbrTargets out;
    out.Color = float4(0.0f);

    // A discarded fragment keeps running to the end of the shader: its texture coordinates still feed
    // the gradients its quad neighbours sample with, and leaving early would blur every masked edge.
    // Point draws cover a square sprite, which rounds off here.
    if (topology == uint(MeshPrimitiveTopology_Point) && length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    if (VelocityOutput) {
        // Discarded fragments write nothing, leaving the motion of whatever surface (or the
        // background) shows through.
        // Each pose projects through its own view: looking through an animated camera moves the view
        // across the shutter too, and a pure camera move is motion like any other. A pose that lands
        // on its camera plane projects to NaN and falls back to no motion.
        const float2 curr_ndc = ProjectToNdc(scene.ViewProj(), in.WorldPosition);
        float2 prev_ndc = ProjectToNdc(scene.PrevViewProj(), in.WorldPosition + in.MotionPrev);
        float2 next_ndc = ProjectToNdc(scene.NextViewProj(), in.WorldPosition + in.MotionNext);
        if (any(isnan(prev_ndc))) prev_ndc = curr_ndc;
        if (any(isnan(next_ndc))) next_ndc = curr_ndc;
        out.Motion = PackScreenMotion(prev_ndc, curr_ndc, next_ndc);
    }

    device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[in.MaterialIndex];
    const float3 world_normal = ShadingWorldNormal(in);
    if (material.DoubleSided == 0u && !IsFrontFacing(scene, world_normal, in.WorldPosition)) discard_fragment();

    float4 base_color = float4(material.BaseColorFactor);
    if (material.BaseColorTexture.Slot != INVALID_MATERIAL_SLOT) base_color *= ctx.SampleTexture(material.BaseColorTexture);
    base_color *= in.VertexColor;
    if (material.AlphaMode == MaterialAlphaMode_Opaque) base_color.a = 1.0f;

    // A point or line vertex without a NORMAL has no surface orientation, so per the glTF spec it renders unlit.
    const bool no_normal = topology != uint(MeshPrimitiveTopology_Triangle) && dot(world_normal, world_normal) < 1e-12f;
    // Unlit fast path. An active debug channel falls through so every per-pixel property is computed.
    if (no_normal || (material.Unlit != 0u && view.DebugChannel == DebugChannel_None)) {
        if (material.AlphaMode == MaterialAlphaMode_Mask) {
            if (base_color.a < material.AlphaCutoff) discard_fragment();
            base_color.a = 1.0f;
        }
        // A no-normal vertex keeps its emissive term, which KHR_materials_unlit drops.
        float3 unlit = base_color.rgb;
        if (no_normal) {
            float3 emissive = float3(material.EmissiveFactor);
            if (material.EmissiveTexture.Slot != INVALID_MATERIAL_SLOT) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
            unlit += emissive;
        }
        out.Color = TransmissionPrepass ? float4(unlit, base_color.a) : float4(unlit * view.Exposure, base_color.a);
        return out;
    }

    const float3 v = normalize(float3(view.CameraPosition) - in.WorldPosition);
    const NormalInfo normal_info = ctx.GetNormalInfo(material);
    const float3 n = normal_info.n;
    const float NdotV = clampedDot(n, v);

    float metallic = material.MetallicFactor;
    float perceptual_roughness = material.RoughnessFactor;
    if (material.MetallicRoughnessTexture.Slot != INVALID_MATERIAL_SLOT) {
        const float4 metallic_roughness = ctx.SampleTexture(material.MetallicRoughnessTexture);
        perceptual_roughness *= metallic_roughness.g;
        metallic *= metallic_roughness.b;
    }
    metallic = clamp(metallic, 0.0f, 1.0f);
    perceptual_roughness = clamp(perceptual_roughness, 0.0f, 1.0f);
    const float alpha_roughness = perceptual_roughness * perceptual_roughness;

    // KHR_materials_sheen
    float3 sheen_color = float3(0.0f);
    float sheen_roughness = 0.0f;
    if (EnableSheen) {
        sheen_color = float3(material.Sheen.ColorFactor);
        if (material.Sheen.ColorTexture.Slot != INVALID_MATERIAL_SLOT) sheen_color *= ctx.SampleTexture(material.Sheen.ColorTexture).rgb;
        sheen_roughness = clamp(material.Sheen.RoughnessFactor, 0.0f, 1.0f);
        if (material.Sheen.RoughnessTexture.Slot != INVALID_MATERIAL_SLOT) sheen_roughness *= ctx.SampleTexture(material.Sheen.RoughnessTexture).a;
    }
    const bool has_sheen = EnableSheen && any(sheen_color > float3(0.0f));

    // KHR_materials_specular: specular_weight modulates dielectric F0 and F90.
    // The defaults reproduce the standard 0.04 F0.
    float specular_weight = material.Specular.Factor;
    if (material.Specular.Texture.Slot != INVALID_MATERIAL_SLOT) specular_weight *= ctx.SampleTexture(material.Specular.Texture).a;
    const float f0_ior_t = (material.Ior - 1.0f) / (material.Ior + 1.0f);
    const float f0_ior = f0_ior_t * f0_ior_t;
    float3 specular_color = float3(material.Specular.ColorFactor);
    if (material.Specular.ColorTexture.Slot != INVALID_MATERIAL_SLOT) specular_color *= ctx.SampleTexture(material.Specular.ColorTexture).rgb;
    const float3 f0_dielectric = min(float3(f0_ior) * specular_color, float3(1.0f));
    const float3 f90_dielectric = float3(specular_weight);

    // KHR_materials_transmission / KHR_materials_dispersion
    float transmission_factor = 0.0f;
    if (EnableTransmission) {
        transmission_factor = material.Transmission.Factor;
        if (material.Transmission.Texture.Slot != INVALID_MATERIAL_SLOT) transmission_factor *= ctx.SampleTexture(material.Transmission.Texture).r;
    }
    // Building the transmission framebuffer drops transmission fragments, preserving opaque geometry behind them.
    if (TransmissionPrepass && transmission_factor > 0.0f) discard_fragment();
    // KHR_materials_diffuse_transmission
    float diffuse_transmission_factor = 0.0f;
    float3 diffuse_transmission_color = float3(0.0f);
    if (EnableDiffuseTrans) {
        diffuse_transmission_factor = material.DiffuseTransmission.Factor;
        if (material.DiffuseTransmission.Texture.Slot != INVALID_MATERIAL_SLOT) diffuse_transmission_factor *= ctx.SampleTexture(material.DiffuseTransmission.Texture).a;
        diffuse_transmission_color = float3(material.DiffuseTransmission.ColorFactor);
        if (material.DiffuseTransmission.ColorTexture.Slot != INVALID_MATERIAL_SLOT) diffuse_transmission_color *= ctx.SampleTexture(material.DiffuseTransmission.ColorTexture).rgb;
    }
    // KHR_materials_volume: ThicknessFactor is model-space, so it multiplies by world scale for Beer's law.
    float world_thickness = material.Volume.ThicknessFactor * in.WorldScale;
    if (material.Volume.ThicknessTexture.Slot != INVALID_MATERIAL_SLOT) world_thickness *= ctx.SampleTexture(material.Volume.ThicknessTexture).g;
    const float3 attenuation_color = float3(material.Volume.AttenuationColor);
    const float attenuation_distance = material.Volume.AttenuationDistance;

    // KHR_materials_clearcoat
    float clearcoat_factor = 0.0f;
    float cc_perceptual_roughness = 0.0f;
    float cc_alpha_roughness = 0.0f;
    float3 n_cc = normal_info.ng;
    float NdotV_cc = 0.0f;
    if (EnableClearcoat) {
        clearcoat_factor = material.Clearcoat.Factor;
        if (material.Clearcoat.Texture.Slot != INVALID_MATERIAL_SLOT) clearcoat_factor *= ctx.SampleTexture(material.Clearcoat.Texture).r;
        cc_perceptual_roughness = clamp(material.Clearcoat.RoughnessFactor, 0.0f, 1.0f);
        if (material.Clearcoat.RoughnessTexture.Slot != INVALID_MATERIAL_SLOT) cc_perceptual_roughness *= ctx.SampleTexture(material.Clearcoat.RoughnessTexture).g;
        cc_perceptual_roughness = clamp(cc_perceptual_roughness, 0.0f, 1.0f);
        cc_alpha_roughness = cc_perceptual_roughness * cc_perceptual_roughness;
        // The clearcoat normal defaults to the geometric normal, and its own normal map overrides it.
        // It uses the same tangent basis as the base material.
        if (material.Clearcoat.NormalTexture.Slot != INVALID_MATERIAL_SLOT) {
            float3 cc_ntex = ctx.SampleTexture(material.Clearcoat.NormalTexture).rgb * 2.0f - float3(1.0f);
            cc_ntex *= float3(material.Clearcoat.NormalScale, material.Clearcoat.NormalScale, 1.0f);
            n_cc = normalize(GetNormalMapTBN(normal_info.t, normal_info.b, normal_info.ng, material.Clearcoat.NormalTexture.UvRotation) * normalize(cc_ntex));
        }
        NdotV_cc = clampedDot(n_cc, v);
    }
    const bool has_clearcoat = EnableClearcoat && clearcoat_factor > 0.0f;

    // KHR_materials_anisotropy
    float anisotropy_strength = 0.0f;
    float2 anisotropy_dir = float2(1.0f, 0.0f);
    float3 anisotropic_t = float3(0.0f);
    float3 anisotropic_b = float3(0.0f);
    if (EnableAnisotropy) {
        anisotropy_strength = material.Anisotropy.Strength;
        // Pre-rotate the tangent-space direction by the material rotation angle.
        anisotropy_dir = float2(cos(material.Anisotropy.Rotation), sin(material.Anisotropy.Rotation));
        if (material.Anisotropy.Texture.Slot != INVALID_MATERIAL_SLOT) {
            const float3 anisotropySample = ctx.SampleTexture(material.Anisotropy.Texture).rgb;
            // The texture's RG encodes direction in [0,1], remapped to [-1,1] then rotated by the material angle.
            const float2 texDir = anisotropySample.xy * 2.0f - float2(1.0f);
            const float2x2 rotMatrix = float2x2(float2(anisotropy_dir.x, anisotropy_dir.y), float2(-anisotropy_dir.y, anisotropy_dir.x));
            anisotropy_dir = normalize(rotMatrix * texDir);
            anisotropy_strength *= anisotropySample.z;
        }
        anisotropy_strength = clamp(anisotropy_strength, 0.0f, 1.0f);
        // World-space anisotropy axes. anisotropic_b uses the geometric normal, matching the reference.
        anisotropic_t = normalize(float3x3(normal_info.t, normal_info.b, normal_info.n) * float3(anisotropy_dir, 0.0f));
        anisotropic_b = cross(normal_info.ng, anisotropic_t);
    }
    const bool has_anisotropy = EnableAnisotropy && anisotropy_strength > 0.0f;

    // KHR_materials_iridescence
    float iridescence_factor = 0.0f;
    float iridescence_thickness = 0.0f;
    float3 iridescence_fresnel_dielectric = float3(0.0f);
    float3 iridescence_fresnel_metallic = float3(0.0f);
    if (EnableIridescence) {
        iridescence_factor = material.Iridescence.Factor;
        if (material.Iridescence.Texture.Slot != INVALID_MATERIAL_SLOT) iridescence_factor *= ctx.SampleTexture(material.Iridescence.Texture).r;
        iridescence_factor = clamp(iridescence_factor, 0.0f, 1.0f);
        iridescence_thickness = material.Iridescence.ThicknessMaximum;
        if (material.Iridescence.ThicknessTexture.Slot != INVALID_MATERIAL_SLOT) {
            const float t = ctx.SampleTexture(material.Iridescence.ThicknessTexture).g;
            iridescence_thickness = mix(material.Iridescence.ThicknessMinimum, material.Iridescence.ThicknessMaximum, t);
        }
        // Iridescence Fresnel is precomputed once at NdotV rather than per light, matching the reference.
        iridescence_fresnel_dielectric = evalIridescence(1.0f, material.Iridescence.Ior, NdotV, iridescence_thickness, f0_dielectric);
        iridescence_fresnel_metallic = evalIridescence(1.0f, material.Iridescence.Ior, NdotV, iridescence_thickness, base_color.rgb);
        if (iridescence_thickness == 0.0f) iridescence_factor = 0.0f;
    }
    const bool has_iridescence = EnableIridescence && iridescence_factor > 0.0f;

    // Hoisted: NdotV is constant across lights.
    const float sheen_lut_ndotv = has_sheen ? albedoSheenScalingLUT(scene, NdotV, sheen_roughness) : 0.0f;

    float3 direct_color = float3(0.0f);
    if (EnablePunctual && view.UseSceneLightsRender != 0u) {
        device const PunctualLight *lights = scene.Lights(view.LightSlot);
        for (uint i = 0u; i < view.LightCount; ++i) {
            const PunctualLight light = lights[i];

            float3 L;
            float3 point_to_light;
            const float3 light_intensity = getLightIntensity(scene, light, in.WorldPosition, L, point_to_light);
            const float3 H = normalize(L + v);
            const float NdotL = clampedDot(n, L);
            const float NdotH = clampedDot(n, H);
            const float VdotH = clampedDot(v, H);

            float3 dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(VdotH));
            const float3 metal_fresnel = F_Schlick(base_color.rgb, float3(1.0f), abs(VdotH));

            float3 l_diffuse = light_intensity * NdotL * BRDF_lambertian(base_color.rgb);
            if (diffuse_transmission_factor > 0.0f) {
                l_diffuse *= (1.0f - diffuse_transmission_factor);
                if (dot(n, L) < 0.0f) {
                    float3 l_diffuse_btdf = light_intensity * clampedDot(-n, L) * BRDF_lambertian(diffuse_transmission_color);
                    const float3 l_mirror = normalize(L + 2.0f * n * dot(-L, n));
                    dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(clampedDot(v, normalize(l_mirror + v))));
                    l_diffuse_btdf = applyVolumeAttenuation(l_diffuse_btdf, world_thickness, attenuation_color, attenuation_distance);
                    l_diffuse += l_diffuse_btdf * diffuse_transmission_factor;
                }
            }
            if (transmission_factor > 0.0f) {
                const float3 transmission_ray = getVolumeTransmissionRay(n, v, world_thickness, material.Ior);
                const float3 transmit_l = safeNormalize(point_to_light - transmission_ray, L);
                float3 l_transmit = light_intensity * getPunctualRadianceTransmission(n, v, transmit_l, alpha_roughness, base_color.rgb, material.Ior);
                l_transmit = applyVolumeAttenuation(l_transmit, length(transmission_ray), attenuation_color, attenuation_distance);
                l_diffuse = mix(l_diffuse, l_transmit, transmission_factor);
            }
            const float3 l_specular = light_intensity * NdotL * (has_anisotropy
                ? BRDF_specularGGXAnisotropy(alpha_roughness, anisotropy_strength, n, v, L, H, anisotropic_t, anisotropic_b)
                : BRDF_specularGGX(alpha_roughness, NdotL, NdotV, NdotH));

            float3 l_metal_brdf = metal_fresnel * l_specular;
            float3 l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);
            if (has_iridescence) {
                l_metal_brdf = mix(l_metal_brdf, l_specular * iridescence_fresnel_metallic, iridescence_factor);
                l_dielectric_brdf = mix(l_dielectric_brdf, mix(l_diffuse, l_specular, iridescence_fresnel_dielectric), iridescence_factor);
            }
            float3 l_color = mix(l_dielectric_brdf, l_metal_brdf, metallic);
            if (has_sheen) {
                const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
                const float l_albedo_sheen_scaling = min(
                    1.0f - max_sheen * sheen_lut_ndotv,
                    1.0f - max_sheen * albedoSheenScalingLUT(scene, NdotL, sheen_roughness));
                l_color = light_intensity * NdotL * BRDF_specularSheen(sheen_color, sheen_roughness, NdotL, NdotV, NdotH)
                    + l_color * l_albedo_sheen_scaling;
            }
            if (has_clearcoat) {
                const float NdotL_cc = clampedDot(n_cc, L);
                const float NdotH_cc = clampedDot(n_cc, H);
                // Intentionally diverges from glTF-Sample-Viewer: Fresnel at the microfacet
                // half-angle (VdotH) rather than a constant view-angle approximation (NdotV_cc).
                const float3 F_cc = F_Schlick(float3(f0_ior), float3(1.0f), abs(VdotH));
                const float3 l_clearcoat = light_intensity * NdotL_cc * F_cc * BRDF_specularGGX(cc_alpha_roughness, NdotL_cc, NdotV_cc, NdotH_cc);
                l_color = l_color * (1.0f - clearcoat_factor * F_cc) + clearcoat_factor * l_clearcoat;
            }
            direct_color += l_color;
        }
    }

    float3 f_diffuse = getDiffuseLight(scene, n) * base_color.rgb;
    if (diffuse_transmission_factor > 0.0f) {
        float3 f_diffuse_transmission = getDiffuseLight(scene, -n) * diffuse_transmission_color;
        f_diffuse_transmission = applyVolumeAttenuation(f_diffuse_transmission, world_thickness, attenuation_color, attenuation_distance);
        f_diffuse = mix(f_diffuse, f_diffuse_transmission, diffuse_transmission_factor);
    }
    if (transmission_factor > 0.0f) {
        const bool real = !TransmissionPrepass
            && view.UseRealTransmission != 0u
            && view.TransmissionFramebufferSamplerSlot != INVALID_MATERIAL_SLOT;
        float3 f_transmission = getVolumeRefraction(scene, n, v, in.WorldPosition, world_thickness, perceptual_roughness, material.Ior, material.Dispersion, real) * base_color.rgb;
        f_transmission = applyVolumeAttenuation(f_transmission, world_thickness, attenuation_color, attenuation_distance);
        f_diffuse = mix(f_diffuse, f_transmission, transmission_factor);
    }
    const float3 f_specular_dielectric = has_anisotropy
        ? getIBLRadianceAnisotropy(scene, n, v, perceptual_roughness, anisotropy_strength, anisotropic_b)
        : getIBLRadianceGGX(scene, n, v, perceptual_roughness);
    const float3 f_specular_metal = f_specular_dielectric;

    const float2 ibl_brdf_f_ab = scene.SampleTex(view.Ibl.BrdfLutSamplerSlot,
                                                 clamp(float2(NdotV, perceptual_roughness), float2(0.0f), float2(1.0f))).rg;
    const float3 f_metal_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, base_color.rgb, 1.0f);
    float3 f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

    const float3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, f0_dielectric, specular_weight);
    float3 f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

    if (has_iridescence) {
        f_metal_brdf_ibl = mix(f_metal_brdf_ibl, f_specular_metal * iridescence_fresnel_metallic, iridescence_factor);
        f_dielectric_brdf_ibl = mix(f_dielectric_brdf_ibl, mix(f_diffuse, f_specular_dielectric, iridescence_fresnel_dielectric), iridescence_factor);
    }
    float3 indirect_color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, metallic);
    if (has_sheen) {
        const float3 f_sheen = getIBLRadianceCharlie(scene, n, v, sheen_roughness, sheen_color);
        const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
        const float albedo_sheen_scaling = 1.0f - max_sheen * sheen_lut_ndotv;
        indirect_color = f_sheen + indirect_color * albedo_sheen_scaling;
    }
    float ao = 1.0f;
    if (material.OcclusionTexture.Slot != INVALID_MATERIAL_SLOT) {
        ao = ctx.SampleTexture(material.OcclusionTexture).r;
        indirect_color *= (1.0f + material.OcclusionStrength * (ao - 1.0f));
    }
    float3 cc_fresnel_ibl = float3(0.0f);
    if (has_clearcoat) {
        // Intentionally diverges from glTF-Sample-Viewer: split-sum BRDF LUT Fresnel
        // (with multi-scattering energy compensation) rather than F_Schlick(F0, F90, NdotV_cc).
        cc_fresnel_ibl = getIBLGGXFresnel(scene, n_cc, v, cc_perceptual_roughness, float3(f0_ior), 1.0f);
        const float3 f_clearcoat_ibl = cc_fresnel_ibl * getIBLRadianceGGX(scene, n_cc, v, cc_perceptual_roughness);
        indirect_color = indirect_color * (1.0f - clearcoat_factor * cc_fresnel_ibl) + clearcoat_factor * f_clearcoat_ibl;
    }

    float3 color = direct_color + indirect_color;
    float3 emissive = float3(material.EmissiveFactor);
    if (material.EmissiveTexture.Slot != INVALID_MATERIAL_SLOT) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
    if (has_clearcoat) emissive *= (1.0f - clearcoat_factor * cc_fresnel_ibl);
    color += emissive;

    if (material.AlphaMode == MaterialAlphaMode_Mask) {
        if (base_color.a < material.AlphaCutoff) discard_fragment();
        base_color.a = 1.0f;
    }

    // An active debug channel replaces the shaded color with the named property. It skips exposure
    // and the face overlay, and the composite passes these values through untransformed.
    if (view.DebugChannel != DebugChannel_None) {
        float3 dbg = float3(0.0f);
        switch (view.DebugChannel) {
            // Generic
            case DebugChannel_UvCoords0: dbg = float3(in.TexCoord0, 0.0f); break;
            case DebugChannel_UvCoords1: dbg = float3(in.TexCoord1, 0.0f); break;
            case DebugChannel_NormalTexture: dbg = (normal_info.ntex + 1.0f) * 0.5f; break;
            case DebugChannel_NormalGeometry: dbg = (normal_info.ng + 1.0f) * 0.5f; break;
            case DebugChannel_NormalShading: dbg = (n + 1.0f) * 0.5f; break;
            case DebugChannel_Tangent: dbg = (normal_info.t + 1.0f) * 0.5f; break;
            case DebugChannel_Bitangent: dbg = (normal_info.b + 1.0f) * 0.5f; break;
            case DebugChannel_TangentW: dbg = float3((in.WorldTangent.w + 1.0f) * 0.5f); break;
            case DebugChannel_Alpha: dbg = float3(base_color.a); break;
            case DebugChannel_Occlusion: dbg = float3(ao); break;
            case DebugChannel_Emissive: dbg = linearTosRGB(emissive); break;
            // Metallic-roughness
            case DebugChannel_BaseColor: dbg = linearTosRGB(base_color.rgb); break;
            case DebugChannel_Metallic: dbg = float3(metallic); break;
            case DebugChannel_Roughness: dbg = float3(perceptual_roughness); break;
            // KHR_materials_clearcoat
            case DebugChannel_ClearcoatFactor: dbg = float3(clearcoat_factor); break;
            case DebugChannel_ClearcoatRoughness: dbg = float3(cc_perceptual_roughness); break;
            case DebugChannel_ClearcoatNormal: dbg = (n_cc + 1.0f) * 0.5f; break;
            // KHR_materials_sheen
            case DebugChannel_SheenColor: dbg = sheen_color; break;
            case DebugChannel_SheenRoughness: dbg = float3(sheen_roughness); break;
            // KHR_materials_specular
            case DebugChannel_SpecularFactor: dbg = float3(specular_weight); break;
            case DebugChannel_SpecularColor: dbg = specular_color; break;
            // KHR_materials_transmission / KHR_materials_volume
            case DebugChannel_TransmissionFactor: dbg = float3(transmission_factor); break;
            case DebugChannel_VolumeThickness: {
                const float denom = material.Volume.ThicknessFactor * in.WorldScale;
                dbg = denom > 0.0f ? float3(world_thickness / denom) : float3(0.0f);
            } break;
            // KHR_materials_diffuse_transmission
            case DebugChannel_DiffuseTransmissionFactor: dbg = linearTosRGB(float3(diffuse_transmission_factor)); break;
            case DebugChannel_DiffuseTransmissionColor: dbg = linearTosRGB(diffuse_transmission_color); break;
            // KHR_materials_iridescence (thickness divided by 1200 nm to match the reference range)
            case DebugChannel_IridescenceFactor: dbg = float3(iridescence_factor); break;
            case DebugChannel_IridescenceThickness: dbg = float3(iridescence_thickness / 1200.0f); break;
            // KHR_materials_anisotropy
            case DebugChannel_AnisotropyStrength: dbg = float3(anisotropy_strength); break;
            case DebugChannel_AnisotropyDirection: dbg = float3((anisotropy_dir + 1.0f) * 0.5f, 0.0f); break;
        }
        out.Color = float4(dbg, base_color.a);
        return out;
    }

    if (!TransmissionPrepass) color *= view.Exposure;

    const uint overlay_flags = in.FaceOverlayFlags & 3u;
    if (overlay_flags != 0u) {
        constant ViewportThemeColors &colors = scene.Theme.Colors;
        const bool is_edit_face = view.InteractionMode == InteractionMode_Edit && view.EditElement == Element_Face;
        const float4 selected = is_edit_face ? float4(colors.FaceSelected) : float4(colors.FaceSelectedIncidental);
        const float3 overlay = (overlay_flags & 2u) != 0u ? mix(selected.rgb, float4(colors.ElementActive).rgb, 0.5f) : selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    out.Color = float4(color, base_color.a);
    return out;
}

fragment PbrTargets PbrMeshletFragment(
    MeshletVertexVaryings meshlet_in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const MeshVaryings in = FromMeshletVertexVaryings(meshlet_in);
    const Scene scene{bindless, view, theme, workspace};
    const PbrContext ctx{scene, in};
    return ShadePbr(
        in, NonTriangleTopology ? meshlet_in.Topology : uint(MeshPrimitiveTopology_Triangle),
        meshlet_in.PointCoord, scene, view, ctx
    );
}

fragment PbrTargets PbrVisibilityFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    DecodedVisibility decoded = DecodeVisibility(quad.Position.xy, visibility, bindless, view, theme, workspace, pc, VelocityOutput);
    if (!decoded.Valid) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, decoded.V};
    ctx.ExplicitGradients = true;
    for (uint set = 0u; set < 4u; ++set) {
        ctx.UvDx[set] = decoded.UvDx[set];
        ctx.UvDy[set] = decoded.UvDy[set];
    }
    return ShadePbr(
        decoded.V, NonTriangleTopology ? decoded.Topology : uint(MeshPrimitiveTopology_Triangle),
        decoded.PointCoord, scene, view, ctx
    );
}

#endif
