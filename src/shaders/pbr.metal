#pragma once
// Adapted from KhronosGroup/glTF-Sample-Renderer.
#include "EeveeGlass.h"
#include "EeveeOpaque.h"
#include "EeveeDiffuse.h"
#include "IridescenceBRDF.metal"
#include "RayGuide.h"
#include "RayWork.h"
#include "RayReuse.h"
#include "RayTracing.h"
#include "RayEnvironmentSampling.h"
#include "Transparency.metal"
#include "VisibilityDecode.metal"
#include "gpu/PbrFeature.h"
#include "ibl.metal"
#include "punctual.metal"
#include "AreaLight.h"
#include "LightCullingLib.h"
constant bool TiledLights [[function_constant(uint(PbrFeature::TiledLights))]];
#include "tonemapping.metal"

constant bool EnableSceneLights [[function_constant(uint(PbrFeature::SceneLights))]];
constant bool EnableAreaLights [[function_constant(uint(PbrFeature::AreaLights))]];
constant bool MultipleShadowSamples [[function_constant(uint(PbrFeature::MultipleShadowSamples))]];
constant bool OpaqueShadows [[function_constant(uint(PbrFeature::OpaqueShadows))]];
constant bool CullShadowBackfaces [[function_constant(uint(PbrFeature::CullShadowBackfaces))]];
constant bool EnableTransmission [[function_constant(uint(PbrFeature::Transmission))]];
constant bool EnableDiffuseTrans [[function_constant(uint(PbrFeature::DiffuseTrans))]];
constant bool EnableClearcoat [[function_constant(uint(PbrFeature::Clearcoat))]];
constant bool EnableSheen [[function_constant(uint(PbrFeature::Sheen))]];
constant bool EnableAnisotropy [[function_constant(uint(PbrFeature::Anisotropy))]];
constant bool EnableIridescence [[function_constant(uint(PbrFeature::Iridescence))]];
constant bool TransmissionPrepass [[function_constant(uint(PbrFeature::TransmissionPrepass))]];
constant bool NonTriangleTopology [[function_constant(uint(PbrFeature::NonTriangleTopology))]];

constant bool EnableOpaqueReceiver [[function_constant(uint(PbrFeature::OpaqueReceiver))]];
constant bool EnableAlphaBlend [[function_constant(uint(PbrFeature::AlphaBlend))]];
constant bool TraceDiffuse [[function_constant(uint(PbrFeature::TraceDiffuse))]];
constant bool PreparePrimary [[function_constant(uint(PbrFeature::PreparePrimary))]];
constant bool TraceReflections [[function_constant(uint(PbrFeature::TraceReflections))]];

// TransmissionPrepass omits transmission materials and exposure to prevent attachment self-sampling.

struct NormalInfo {
    float3 ng;
    float3 t;
    float3 b;
    float3 n;
    float3 ntex;
};


inline float clampedDot(float3 x, float3 y) { return clamp(dot(x, y), 0.0f, 1.0f); }

// An attenuation distance of zero or less means no attenuation.
inline float3 applyVolumeAttenuation(float3 radiance, float dist, float3 attenuation_color, float attenuation_distance) {
    if (attenuation_distance <= 0.0f) return radiance;
    return pow(attenuation_color, float3(dist / attenuation_distance)) * radiance;
}

// No Fresnel term, transmission D*V only, per the glTF Sample Renderer.
inline float3 getPunctualRadianceTransmission(float3 n, float3 v, float3 l, float alpha_roughness, float3 baseColor, float ior) {
    const float tr = applyIorToRoughness(alpha_roughness, ior);
    const float3 l_mirror = normalize(l + 2.0f * n * dot(-l, n));
    const float3 h = normalize(l_mirror + v);
    return baseColor * D_GGX(clampedDot(n, h), tr) * V_GGX(clampedDot(n, l_mirror), clampedDot(n, v), tr);
}

// The caller subtracts this from point_to_light so the BTDF sees the light direction at the exit point.
inline float3 getVolumeTransmissionRay(float3 n, float3 v, float world_thickness, float ior) {
    return normalize(refract(-v, normalize(n), 1.0f / ior)) * world_thickness;
}

// Rotates the tangent frame by the KHR_texture_transform UV rotation to align sampled normal-map components.
inline float3x3 GetNormalMapTBN(float3 t, float3 b, float3 n, float uv_rotation) {
    const float c = cos(uv_rotation);
    const float s = sin(uv_rotation);
    return float3x3(c * t - s * b, s * t + c * b, n);
}

struct RayReuseSample {
    float4 DirectionPdf;
    float4 IncidentDistance;
};

#ifdef FILTER_AREA_SHADOWS
#include "RayShadowNeighborhood.h"
#endif
struct PbrContext {
    Scene S;
    const thread MeshVaryings &In;
    float2 UvDx[4];
    float2 UvDy[4];
    bool ExplicitGradients;
    bool Secondary, RayTracing, Opaque;
    float3 Outgoing, PositionDx, PositionDy, GeometricNormal;
    float Footprint, GlassFraction;
    uint SampleOffset, SampleCount;
    thread RaySpatialReuse *Reuse;
    thread RayReuseSample *Sample;
    thread eevee_opaque::Closure *OpaqueClosure;
    const thread RayMediumState *Media;
#ifdef FILTER_AREA_SHADOWS
    ShadowNeighborhood ShadowNeighbors;
#endif
    texture2d_array<float> LightLut;

    void SetSurface(const thread DecodedSurface &decoded) {
        ExplicitGradients = true;
        GeometricNormal = decoded.GeometricNormal;
        PositionDx = decoded.PositionDx;
        PositionDy = decoded.PositionDy;
        for (uint set = 0; set < 4; ++set) {
            UvDx[set] = decoded.UvDx[set];
            UvDy[set] = decoded.UvDy[set];
        }
    }

    float3 ViewDirection() const { return Secondary ? Outgoing : RayTracing && S.View.ScreenPixelScale < 0.0f ? transpose(S.View.ViewRotation.Unpack()) * float3(0, 0, 1) :
                                                                                                                normalize(float3(S.View.CameraPosition) - In.WorldPosition); }

    float2 GetUv(TextureInfo tex) const {
        return ApplyUvTransform(RayUv(In, tex.TexCoord), float2(tex.UvOffset), float2(tex.UvScale), tex.UvRotation);
    }
    float4 SampleTexture(TextureInfo tex) const {
        const float2 uv = GetUv(tex);
        if (!ExplicitGradients) return S.SampleTex(tex.Slot, uv);
        const uint set = min(tex.TexCoord, 3u);
        const float2 scale = float2(tex.UvScale);
        const float footprint = Secondary ? Footprint : 1.0f;
        return S.SampleTexGrad(tex.Slot, uv, TransformUvGradient(UvDx[set] * footprint, scale, tex.UvRotation), TransformUvGradient(UvDy[set] * footprint, scale, tex.UvRotation));
    }

    float2 MetallicRoughness(device const PBRMaterial &m) const {
        float2 values(m.MetallicFactor, m.RoughnessFactor);
        if (m.MetallicRoughnessTexture.Slot != InvalidSlot) values *= SampleTexture(m.MetallicRoughnessTexture).bg;
        return values;
    }

    float4 SpecularColorStrength(device const PBRMaterial &m) const {
        float4 specular(float3(m.Specular.ColorFactor), m.Specular.Factor);
        if (m.Specular.ColorTexture.Slot != InvalidSlot) specular.rgb *= SampleTexture(m.Specular.ColorTexture).rgb;
        if (m.Specular.Texture.Slot != InvalidSlot) specular.a *= SampleTexture(m.Specular.Texture).a;
        return specular;
    }

    template<bool Raster> NormalInfo GetNormalInfo(device const PBRMaterial &material) const {
        const float2 uv = RayUv(In, material.NormalTexture.TexCoord);
        float3 ng = normalize(ShadingWorldNormal(In));
        float3 t;
        float3 b;
        if (length(In.WorldTangent.xyz) > 1e-8f) {
            t = normalize(In.WorldTangent.xyz - ng * dot(ng, In.WorldTangent.xyz));
            b = normalize(cross(ng, t) * In.WorldTangent.w);
        } else {
            float3 pos_dx = PositionDx, pos_dy = PositionDy;
            float2 uv_dx = UvDx[min(material.NormalTexture.TexCoord, 3u)];
            float2 uv_dy = UvDy[min(material.NormalTexture.TexCoord, 3u)];
            if constexpr (Raster) {
                pos_dx = dfdx(In.WorldPosition);
                pos_dy = dfdy(In.WorldPosition);
                uv_dx = dfdx(uv);
                uv_dy = dfdy(uv);
            }
            const float uv_scale = max(max(abs(uv_dx.x), abs(uv_dx.y)), max(abs(uv_dy.x), abs(uv_dy.y)));
            if (uv_scale > 0.0f) {
                uv_dx /= uv_scale;
                uv_dy /= uv_scale;
            }
            const float det = uv_dx.x * uv_dy.y - uv_dy.x * uv_dx.y;
            const float3 derivative_t = sign(det) * (uv_dy.y * pos_dx - uv_dx.y * pos_dy);
            const float3 derivative_b = sign(det) * (uv_dx.x * pos_dy - uv_dy.x * pos_dx);
            const float3 projected_t = derivative_t - ng * dot(ng, derivative_t);
            if (det != 0.0f && dot(projected_t, projected_t) > 0.0f) {
                t = normalize(projected_t);
                const float handedness = dot(cross(ng, t), derivative_b) < 0.0f ? -1.0f : 1.0f;
                b = cross(ng, t) * handedness;
            } else {
                const float3 axis = abs(ng.x) < abs(ng.y) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
                t = normalize(cross(axis, ng));
                b = cross(ng, t);
            }
        }

        if (Raster ? !IsFrontFacing(S, ng, In.WorldPosition) : dot(ng, ViewDirection()) < 0.0f) {
            t *= -1.0f;
            b *= -1.0f;
            ng *= -1.0f;
        }

        if (material.NormalTexture.Slot != InvalidSlot) {
            float3 ntex = SampleTexture(material.NormalTexture).rgb * 2.0f - float3(1.0f);
            ntex *= float3(material.NormalScale, material.NormalScale, 1.0f);
            ntex = normalize(ntex);
            return {.ng = ng, .t = t, .b = b, .n = normalize(GetNormalMapTBN(t, b, ng, material.NormalTexture.UvRotation) * ntex), .ntex = ntex};
        }
        return {.ng = ng, .t = t, .b = b, .n = ng, .ntex = float3(0.0f, 0.0f, 1.0f)};
    }
};

// Keep each transmissive query outside the emitter-sampling loop's compiled body.
// Keep these call boundaries: inlining reproduced glass speckles under instrumentation.
template<bool Glass = true>
__attribute__((noinline)) float3 AreaRayTransmittance(const thread Scene &scene, const thread RayGeometry &rays,
    float3 origin, float3 direction, float maximum, bool transmit_glass, RayMediumState media = {}) {
    return RayTransmittance<Glass>(scene, rays, origin, direction, maximum, transmit_glass, media);
}

__attribute__((noinline)) float3 AreaRayTransmittanceEmpty(const thread Scene &scene, const thread RayGeometry &rays,
    float3 origin, float3 direction, float maximum, bool transmit_glass) {
    return RayTransmittance(scene, rays, origin, direction, maximum, transmit_glass);
}

__attribute__((noinline)) bool AreaRayOccluded(const thread Scene &scene, const thread RayGeometry &rays,
    float3 origin, float3 direction, float maximum) {
    return RayOccluded(scene, rays, origin, direction, maximum, OpaqueShadows, CullShadowBackfaces);
}

#ifdef BOUNDED_AREA_SHADOWS
#include "AreaShadowConnection.h"
#endif
inline float3 AreaShadowVisibility(const thread Scene &scene, const thread RayScene &rays,
    float3 position, float3 normal, const thread AreaEmitter &emitter, float2 noise, const thread RayMediumState &media) {
    const RayGeometry geometry = rays.Geometry();
    const uint count = max(scene.View.RayShadowSamples, 1u);
#ifdef BOUNDED_AREA_SHADOWS
    device const auto *lights = &BindlessBuffer(PreparedAreaLight, scene.B.Buffer, scene.View.RayAreaLightSlot)[0];
    const AreaShadowBounds bounds = *reinterpret_cast<device const AreaShadowBounds *>(lights + scene.View.LightCount);
    const bool clear = media.Count == 0u && AreaConnectionClear(position, emitter.Center, abs(emitter.X) + abs(emitter.Y), bounds);
#endif
    float3 visibility(0);
    for (uint sample_index = 0; sample_index < count; ++sample_index) {
        const float2 u = fract(noise + eevee_denoise::hammersley(sample_index, count));
        const float2 sample = eevee_ltc::sample_shape(u, emitter.Rectangle);
        const float3 target = emitter.Center + (emitter.X * sample.x + emitter.Y * sample.y);
        const float3 direction = normalize(target);
        const float3 origin = RayOrigin(position, normal, direction);
        const float maximum = length(target) - distance(origin, position);
#ifdef BOUNDED_AREA_SHADOWS
        if (clear) {
            visibility += float(!RayOpaqueOccluded(geometry.Acceleration, origin, direction, maximum, bounds.CullBackfaces != 0u));
            continue;
        }
#endif
        if (EnableTransmission || EnableAlphaBlend) {
            visibility += EnableTransmission ? (media.Count == 0u ? AreaRayTransmittanceEmpty(scene, geometry, origin, direction, maximum, scene.View.RayTransmissionShadows) : AreaRayTransmittance(scene, geometry, origin, direction, maximum, scene.View.RayTransmissionShadows, media)) : AreaRayTransmittance<false>(scene, geometry, origin, direction, maximum, false);
        } else visibility += float(!AreaRayOccluded(scene, geometry, origin, direction, maximum));
    }
    return visibility / float(count);
}

inline eevee_glass::Closure PrepareGlass(const thread PbrContext &ctx, const thread RayScene &rt) {
    device const PBRMaterial &m = ctx.S.Materials(ctx.S.View.MaterialSlot)[ctx.In.MaterialIndex];
    eevee_glass::Closure cl{};
    cl.N = ctx.GetNormalInfo<false>(m).n;
    cl.V = ctx.ViewDirection();
    cl.Position = ctx.In.WorldPosition;
    cl.Ior = max(m.Ior, 1.0f);
    const float2 mr = saturate(ctx.MetallicRoughness(m));
    cl.Roughness = mr.y;
    cl.Fraction = m.Transmission.Factor;
    if (m.Transmission.Texture.Slot != InvalidSlot) cl.Fraction *= ctx.SampleTexture(m.Transmission.Texture).r;
    cl.Fraction = saturate(cl.Fraction) * (1.0f - mr.x);
    cl.Thickness = m.Volume.ThicknessFactor * ctx.In.WorldScale;
    if (m.Volume.ThicknessTexture.Slot != InvalidSlot) cl.Thickness *= ctx.SampleTexture(m.Volume.ThicknessTexture).g;
    if (m.Volume.Shape == ThicknessShape::Slab) cl.Thickness = -cl.Thickness;
    float3 tint = float3(m.BaseColorFactor.rgb) * ctx.In.VertexColor.rgb;
    if (m.BaseColorTexture.Slot != InvalidSlot) tint *= ctx.SampleTexture(m.BaseColorTexture).rgb;
    tint = max(tint, 0.0f);
    const float4 specular = ctx.SpecularColorStrength(m);
    const float nv = saturate(dot(cl.N, cl.V));
    const float f0 = eevee_glass::square((cl.Ior - 1.0f) / (cl.Ior + 1.0f));
    eevee_glass::weights(rt.Lut, nv, cl.Roughness, cl.Ior, cl.Thickness == 0.0f ? float3(1) : sqrt(tint), saturate(specular.rgb * specular.a * f0), float3(1), cl.Reflection, cl.Transmission);
    if (cl.Thickness == 0.0f) {
        const float3 c = pow(tint, float3(cl.Ior / sqrt(cl.Ior * cl.Ior - (1.0f - nv * nv))));
        cl.Transmission = c * cl.Transmission * cl.Transmission / max(1.0f - cl.Reflection * cl.Reflection * c * c, 1e-8f);
        cl.Reflection *= 1.0f + cl.Transmission * c;
    }
    cl.Reflection *= cl.Fraction;
    cl.Transmission *= cl.Fraction;
    if (cl.Thickness != 0.0f) {
        cl.Transmission *= cl.Transmission;
        cl.Transmission = applyVolumeAttenuation(cl.Transmission, max(0.0f, cl.transmission_distance()), float3(m.Volume.AttenuationColor), m.Volume.AttenuationDistance);
    }
    return cl;
}

struct RayClosureWeights {
    float4 Reflection, Transmission, GlassShape;
};
inline RayClosureWeights GlassWeights(const thread eevee_glass::Closure &cl) {
    return {float4(cl.Reflection, cl.Roughness), float4(cl.Transmission, cl.roughness(1u)), float4(EncodeRayNormal(cl.N), cl.Roughness, cl.Thickness)};
}

enum class PbrMode : uint { Raster,
                            Secondary,
                            Direct,
                            Indirect,
                            GlassDirect,
                            GlassSurface,
                            Reuse };
inline float3 TraceRayRadiance(const thread Scene &scene, const thread RayScene &rt, DecodedSurface surface, float3 direction, float footprint, const thread RayMediumState *initial_media = nullptr, float initial_distance = 0.0f, thread float *camera_hole = nullptr);

template<PbrMode Mode = PbrMode::Raster, bool Fragment = Mode == PbrMode::Raster>
inline float4 ShadePbr(
    MeshVaryings in, uint topology, float2 point_coord, const thread Scene &scene,
    constant SceneViewUBO &view, const thread PbrContext &ctx, const thread RayScene *rt = nullptr, thread float3 *traced_radiance = nullptr, thread float *surface_roughness = nullptr, thread float4 *hit_guide = nullptr
) {
    if (surface_roughness) *surface_roughness = -1.0f;
    // Discarded lanes continue so neighboring texture-gradient calculations remain valid at masked edges.
    if constexpr (Fragment)
        if (topology == uint(MeshPrimitiveTopology::Point) && length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();

    if (topology != uint(MeshPrimitiveTopology::Triangle) && in.Color.a > 0.0f && view.DebugChannel == DebugChannel::None) {
        return float4(in.Color.rgb, 1.0f);
    }

    device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[in.MaterialIndex];
    const float3 world_normal = ShadingWorldNormal(in);
    if constexpr (Fragment)
        if (!ctx.Secondary && CullBackfaces(scene.View, material.DoubleSided) && !IsFrontFacing(scene, world_normal, in.WorldPosition)) discard_fragment();

    float4 base_color = float4(material.BaseColorFactor);
    if (material.BaseColorTexture.Slot != InvalidSlot) base_color *= ctx.SampleTexture(material.BaseColorTexture);
    base_color *= in.VertexColor;
    if (material.AlphaMode == MaterialAlphaMode::Opaque) base_color.a = 1.0f;

    const bool no_normal = topology != uint(MeshPrimitiveTopology::Triangle) && dot(world_normal, world_normal) < 1e-12f;
    if (no_normal || (material.Unlit != 0u && view.DebugChannel == DebugChannel::None)) {
        if (material.AlphaMode == MaterialAlphaMode::Mask) {
            if constexpr (Fragment)
                if (!ctx.Secondary && base_color.a < material.AlphaCutoff) discard_fragment();
            base_color.a = 1.0f;
        }
        float3 unlit = base_color.rgb;
        if (no_normal) {
            float3 emissive = float3(material.EmissiveFactor) * material.EmissiveStrength;
            if (material.EmissiveTexture.Slot != InvalidSlot) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
            unlit += emissive;
        }
        return TransmissionPrepass || ctx.Secondary ? float4(unlit, base_color.a) : float4(unlit * view.Exposure, base_color.a);
    }

    const float3 v = ctx.ViewDirection();
    const NormalInfo normal_info = ctx.GetNormalInfo<Fragment>(material);
    const float3 n = normal_info.n;
    const float NdotV = clampedDot(n, v);

    const float2 mr = clamp(ctx.MetallicRoughness(material), 0.0f, 1.0f);
    const float metallic = mr.x, perceptual_roughness = mr.y;
    const float alpha_roughness = Mode != PbrMode::Raster ? max(0.001f, perceptual_roughness * perceptual_roughness) : perceptual_roughness * perceptual_roughness;

    float3 sheen_color = float3(0.0f);
    float sheen_roughness = 0.0f;
    if (EnableSheen) {
        sheen_color = float3(material.Sheen.ColorFactor);
        if (material.Sheen.ColorTexture.Slot != InvalidSlot) sheen_color *= ctx.SampleTexture(material.Sheen.ColorTexture).rgb;
        sheen_roughness = clamp(material.Sheen.RoughnessFactor, 0.0f, 1.0f);
        if (material.Sheen.RoughnessTexture.Slot != InvalidSlot) sheen_roughness *= ctx.SampleTexture(material.Sheen.RoughnessTexture).a;
    }
    const bool has_sheen = EnableSheen && any(sheen_color > float3(0.0f));

    const float4 specular = ctx.SpecularColorStrength(material);
    const float specular_weight = specular.a;
    const float f0_ior_t = (material.Ior - 1.0f) / (material.Ior + 1.0f);
    const float f0_ior = f0_ior_t * f0_ior_t;
    const float3 specular_color = specular.rgb;
    const float3 f0_dielectric = min(float3(f0_ior) * specular_color, float3(1.0f));
    const float3 f90_dielectric = float3(specular_weight);

    float transmission_factor = 0.0f;
    if (EnableTransmission && !ctx.Opaque) {
        transmission_factor = material.Transmission.Factor;
        if (material.Transmission.Texture.Slot != InvalidSlot) transmission_factor *= ctx.SampleTexture(material.Transmission.Texture).r;
    }
    // The transmission prepass drops transmissive fragments so it captures the opaque geometry behind them.
    if constexpr (Fragment)
        if (TransmissionPrepass && transmission_factor > 0.0f) discard_fragment();
    float diffuse_transmission_factor = 0.0f;
    float3 diffuse_transmission_color = float3(0.0f);
    if (EnableDiffuseTrans) {
        diffuse_transmission_factor = material.DiffuseTransmission.Factor;
        if (material.DiffuseTransmission.Texture.Slot != InvalidSlot) diffuse_transmission_factor *= ctx.SampleTexture(material.DiffuseTransmission.Texture).a;
        diffuse_transmission_color = float3(material.DiffuseTransmission.ColorFactor);
        if (material.DiffuseTransmission.ColorTexture.Slot != InvalidSlot) diffuse_transmission_color *= ctx.SampleTexture(material.DiffuseTransmission.ColorTexture).rgb;
    }
    // KHR_materials_volume thickness converts from model to world space for Beer's law.
    float world_thickness = material.Volume.ThicknessFactor * in.WorldScale;
    if (material.Volume.ThicknessTexture.Slot != InvalidSlot) world_thickness *= ctx.SampleTexture(material.Volume.ThicknessTexture).g;
    const float3 attenuation_color = float3(material.Volume.AttenuationColor);
    const float attenuation_distance = material.Volume.AttenuationDistance;

    float clearcoat_factor = 0.0f;
    float cc_perceptual_roughness = 0.0f;
    float cc_alpha_roughness = 0.0f;
    float3 n_cc = normal_info.ng;
    float NdotV_cc = 0.0f;
    if (EnableClearcoat) {
        clearcoat_factor = material.Clearcoat.Factor;
        if (material.Clearcoat.Texture.Slot != InvalidSlot) clearcoat_factor *= ctx.SampleTexture(material.Clearcoat.Texture).r;
        cc_perceptual_roughness = clamp(material.Clearcoat.RoughnessFactor, 0.0f, 1.0f);
        if (material.Clearcoat.RoughnessTexture.Slot != InvalidSlot) cc_perceptual_roughness *= ctx.SampleTexture(material.Clearcoat.RoughnessTexture).g;
        cc_perceptual_roughness = clamp(cc_perceptual_roughness, 0.0f, 1.0f);
        cc_alpha_roughness = max(Mode != PbrMode::Raster ? 0.001f : 0.0f, cc_perceptual_roughness * cc_perceptual_roughness);
        if (material.Clearcoat.NormalTexture.Slot != InvalidSlot) {
            float3 cc_ntex = ctx.SampleTexture(material.Clearcoat.NormalTexture).rgb * 2.0f - float3(1.0f);
            cc_ntex *= float3(material.Clearcoat.NormalScale, material.Clearcoat.NormalScale, 1.0f);
            n_cc = normalize(GetNormalMapTBN(normal_info.t, normal_info.b, normal_info.ng, material.Clearcoat.NormalTexture.UvRotation) * normalize(cc_ntex));
        }
        NdotV_cc = clampedDot(n_cc, v);
    }
    const bool has_clearcoat = EnableClearcoat && clearcoat_factor > 0.0f;

    float anisotropy_strength = 0.0f;
    float2 anisotropy_dir = float2(1.0f, 0.0f);
    float3 anisotropic_t = float3(0.0f);
    float3 anisotropic_b = float3(0.0f);
    if (EnableAnisotropy) {
        anisotropy_strength = material.Anisotropy.Strength;
        anisotropy_dir = float2(cos(material.Anisotropy.Rotation), sin(material.Anisotropy.Rotation));
        if (material.Anisotropy.Texture.Slot != InvalidSlot) {
            const float3 anisotropySample = ctx.SampleTexture(material.Anisotropy.Texture).rgb;
            const float2 texDir = anisotropySample.xy * 2.0f - float2(1.0f);
            const float2x2 rotMatrix = float2x2(float2(anisotropy_dir.x, anisotropy_dir.y), float2(-anisotropy_dir.y, anisotropy_dir.x));
            anisotropy_dir = normalize(rotMatrix * texDir);
            anisotropy_strength *= anisotropySample.z;
        }
        anisotropy_strength = clamp(anisotropy_strength, 0.0f, 1.0f);
        // anisotropic_b uses the geometric normal to match the reference implementation.
        anisotropic_t = normalize(float3x3(normal_info.t, normal_info.b, normal_info.n) * float3(anisotropy_dir, 0.0f));
        anisotropic_b = cross(normal_info.ng, anisotropic_t);
    }
    const bool has_anisotropy = EnableAnisotropy && anisotropy_strength > 0.0f;

    float iridescence_factor = 0.0f;
    float iridescence_thickness = 0.0f;
    float3 iridescence_fresnel_dielectric = float3(0.0f);
    float3 iridescence_fresnel_metallic = float3(0.0f);
    if (EnableIridescence) {
        iridescence_factor = material.Iridescence.Factor;
        if (material.Iridescence.Texture.Slot != InvalidSlot) iridescence_factor *= ctx.SampleTexture(material.Iridescence.Texture).r;
        iridescence_factor = clamp(iridescence_factor, 0.0f, 1.0f);
        iridescence_thickness = material.Iridescence.ThicknessMaximum;
        if (material.Iridescence.ThicknessTexture.Slot != InvalidSlot) {
            const float t = ctx.SampleTexture(material.Iridescence.ThicknessTexture).g;
            iridescence_thickness = mix(material.Iridescence.ThicknessMinimum, material.Iridescence.ThicknessMaximum, t);
        }
        // Iridescence Fresnel evaluates once at NdotV to match the reference implementation.
        iridescence_fresnel_dielectric = evalIridescence(1.0f, material.Iridescence.Ior, NdotV, iridescence_thickness, f0_dielectric);
        iridescence_fresnel_metallic = evalIridescence(1.0f, material.Iridescence.Ior, NdotV, iridescence_thickness, base_color.rgb);
        if (iridescence_thickness == 0.0f) iridescence_factor = 0.0f;
    }
    const bool has_iridescence = EnableIridescence && iridescence_factor > 0.0f;

    const float sheen_lut_ndotv = has_sheen ? albedoSheenScalingLUT(scene, NdotV, sheen_roughness) : 0.0f;

    const float trace_roughness = has_clearcoat ? min(perceptual_roughness, cc_perceptual_roughness) : perceptual_roughness;
    if (surface_roughness) *surface_roughness = trace_roughness;
    const float reflection_weight = Mode == PbrMode::Secondary || Mode == PbrMode::GlassSurface || Mode == PbrMode::Raster ? 1.0f : TraceReflections ? RayReflectionWeight(trace_roughness, view.RayMaxRoughness) :
                                                                                                                      0.0f;
    constexpr bool indirect = Mode == PbrMode::Indirect || Mode == PbrMode::Reuse;
    // EEVEE prepares light-independent closure data once per surface.
    texture2d_array<float> area_lut = ctx.LightLut;
    if constexpr (Mode != PbrMode::Raster) area_lut = rt->Lut;
    eevee_ltc::LTCData area_diffuse_data{}, area_specular_data{};
    float3 area_dielectric_fresnel{}, area_metal_fresnel{};
    if (EnableAreaLights && EnableSceneLights && view.UseSceneLightsRender && view.LightCount && !indirect) {
        area_diffuse_data = eevee_ltc::diffuse(n, dot(n, v) < 0.0f ? -v : v);
        area_specular_data = eevee_ltc::ggx(area_lut, n, v, perceptual_roughness);
        const float2 energy = eevee_glass::lut2(area_lut, float2(perceptual_roughness, sqrt(saturate(1.0f - NdotV))), 1u).xy;
        area_dielectric_fresnel = eevee_opaque::fresnel(f0_dielectric * specular_weight, f90_dielectric, energy);
        area_metal_fresnel = eevee_opaque::fresnel(base_color.rgb, float3(1), energy);
    }
    float3 direct_color = float3(0.0f), ray_color = float3(0.0f);
    const uint light_count = !indirect && EnableSceneLights && view.UseSceneLightsRender != 0u ? view.LightCount : 0u;
    const uint ray_count = (indirect && (reflection_weight > 0.0f || TraceDiffuse)) ? (Mode == PbrMode::Reuse ? ctx.Reuse->Count : max(ctx.SampleCount ? ctx.SampleCount : view.RaySamples, 1u)) : 0u;
    float2 shadow_blue_noise{};
    if constexpr (Mode == PbrMode::Direct) {
        if (!MultipleShadowSamples && !EnableTransmission && !EnableAlphaBlend && EnableAreaLights && light_count && !ctx.Secondary) shadow_blue_noise = area_lut.read(uint2(in.Position.xy) % 64u, 0u).xy;
    }
    uint rng = RayHash(as_type<uint>(in.WorldPosition.x) ^ RayHash(as_type<uint>(in.WorldPosition.y)) ^ RayHash(as_type<uint>(in.WorldPosition.z))) ^ RayHash(view.RayFrame + ctx.SampleOffset);
    {
        device const LightRecord *lights = scene.Lights(view.LightSlot);
        auto light_iterator = TiledLightIterator::make(scene, in.WorldPosition, light_count, TiledLights);
        for (uint i = light_iterator.next(TiledLights); i < light_count + ray_count; i = i < light_count ? light_iterator.next(TiledLights) : i + 1u) {
            float reuse_pdf = 0.0f;
            float3 L{}, point_to_light{}, light_intensity{};
            bool area = false;
            AreaEmitter emitter{};
            if constexpr (!indirect) {
                const LightRecord light = lights[i];
                area = EnableAreaLights && IsAreaLight(light.Type);
                if (area) {
                    if (!PrepareAreaEmitter(scene, light, in.WorldPosition, emitter, i)) continue;
                    point_to_light = emitter.Center;
                    L = normalize(point_to_light);
                    light_intensity = emitter.Radiance;
                } else light_intensity = getLightIntensity(scene, light, in.WorldPosition, L, point_to_light);
                if constexpr (Mode != PbrMode::Raster) {
                    // Reject zero-contribution punctual lights before tracing their visibility.
                    if (all(light_intensity == float3(0)) ||
                        (!area && dot(n, L) <= 0.0f && (!has_clearcoat || dot(n_cc, L) <= 0.0f) &&
                         diffuse_transmission_factor == 0.0f && transmission_factor == 0.0f)) continue;
                    float2 noise{};
                    if (area) {
                        // EEVEE combines a per-pixel blue-noise tile with a temporal sequence.
                        const float2 rotation = emitter.NoiseRotation;
                        noise = ctx.Secondary ? float2(RayRandom(rng), RayRandom(rng)) : fract(((Mode != PbrMode::Direct || MultipleShadowSamples || EnableTransmission || EnableAlphaBlend) ? area_lut.read(uint2(in.Position.xy) % 64u, 0u).xy : shadow_blue_noise) + rotation + RaySampleOffset(view.RayFrame + ctx.SampleOffset));
                    }
                    if (MultipleShadowSamples && area) {
                        RayMediumState media{};
                        if (ctx.Media) media = *ctx.Media;
                        light_intensity *= AreaShadowVisibility(scene, *rt, in.WorldPosition, ctx.GeometricNormal, emitter, noise, media);
                    } else {
                        float3 shadow_vector = point_to_light;
                        if (area) {
                            const float2 sample = eevee_ltc::sample_shape(noise, emitter.Rectangle);
                            shadow_vector += emitter.X * sample.x + emitter.Y * sample.y;
                        }
                        const float3 shadow_direction = area ? normalize(shadow_vector) : L;
                        const float3 origin = RayOrigin(in.WorldPosition, ctx.GeometricNormal, shadow_direction);
                        const float maximum = light.Type == LightType::Directional ? INFINITY : length(shadow_vector) - distance(origin, in.WorldPosition);
                        if (EnableTransmission || EnableAlphaBlend) {
                            RayMediumState media{};
                            if (ctx.Media) media = *ctx.Media;
                            light_intensity *= EnableTransmission ? RayTransmittance(scene, rt->Geometry(), origin, shadow_direction, maximum, view.RayTransmissionShadows, media) : RayTransmittance<false>(scene, rt->Geometry(), origin, shadow_direction, maximum, false);
                        }
#ifdef FILTER_AREA_SHADOWS
                        else {
                            const bool lit = !RayOccluded(scene, rt->Geometry(), origin, shadow_direction, maximum, OpaqueShadows, CullShadowBackfaces);
                            if (Mode == PbrMode::Direct && area)
                                light_intensity *= ctx.ShadowNeighbors.visibility(lit);
                            else if (!lit) light_intensity = float3(0);
                        }
#else
                        else if (RayOccluded(scene, rt->Geometry(), origin, shadow_direction, maximum, OpaqueShadows, CullShadowBackfaces)) light_intensity = float3(0);
#endif
                    }
                }
            } else {
                const float3x3 frame = RayFrame(n);
                const float3 local_v = transpose(frame) * v;
                const float2 u = float2(RayRandom(rng), RayRandom(rng));
                const float component = RayRandom(rng);
                const bool diffuse_only = reflection_weight == 0.0f;
                const float coat_weight = !diffuse_only && has_clearcoat ? 0.25f * clearcoat_factor : 0.0f;
                const float cosine_weight = diffuse_only ? 1.0f : max(has_sheen ? 0.25f : 0.0f, TraceDiffuse ? 0.5f * (1.0f - metallic) : 0.0f);
                const float base_weight = 1.0f - cosine_weight - coat_weight;
                if constexpr (Mode == PbrMode::Reuse) {
                    if (!ctx.Reuse->Sample(i, L)) continue;
                } else if (component < cosine_weight) {
                    L = frame * eevee_diffuse::sample(u, TraceDiffuse && component < cosine_weight * diffuse_transmission_factor);
                } else if (component < 1.0f - coat_weight) {
                    L = frame * SampleRayGGX(local_v, alpha_roughness, u);
                } else {
                    const float3x3 coat_frame = RayFrame(n_cc);
                    L = coat_frame * SampleRayGGX(transpose(coat_frame) * v, max(cc_alpha_roughness, 0.001f), u);
                }
                const float environment_weight = view.EnvIntensity > 0.0f && (cosine_weight > 0.0f || perceptual_roughness > 0.15f) ? 0.5f : 0.0f;
                if (Mode != PbrMode::Reuse && RayRandom(rng) < environment_weight) L = transpose(view.EnvRotation.Unpack()) * SampleRayEnvironment(rt->Environment, float2(RayRandom(rng), RayRandom(rng)));
                const float bsdf_pdf = cosine_weight * eevee_diffuse::pdf(dot(n, L), TraceDiffuse ? diffuse_transmission_factor : 0.0f) + (base_weight > 0.0f ? base_weight * RayGgxPdf(n, v, L, alpha_roughness) : 0.0f) + (coat_weight > 0.0f ? coat_weight * RayGgxPdf(n_cc, v, L, max(cc_alpha_roughness, 0.001f)) : 0.0f);
                const float pdf = Mode == PbrMode::Reuse ? bsdf_pdf : (1.0f - environment_weight) * bsdf_pdf + environment_weight * RayEnvironmentPdf(rt->Environment, view.EnvRotation.Unpack() * L);
                reuse_pdf = pdf;
                const bool transmitted = TraceDiffuse && diffuse_transmission_factor > 0.0f && dot(n, L) < 0.0f;
                const float geometric_side = dot(ctx.GeometricNormal, L) * dot(ctx.GeometricNormal, v);
                if (pdf <= 0.0f || (transmitted ? -geometric_side : geometric_side) <= 0.0f) continue;
                light_intensity = float3(0);
                if constexpr (Mode == PbrMode::Reuse) light_intensity = ctx.Reuse->Radiance;
                else {
                    const DecodedSurface hit = TraceSurface(scene, *rt, RayOrigin(in.WorldPosition, ctx.GeometricNormal, L), L);
                    if (hit_guide && i == 0u) *hit_guide = hit.Valid ? float4(EncodeRayNormal(hit.GeometricNormal), distance(in.WorldPosition, hit.V.WorldPosition), float(hit.V.MaterialIndex)) : float4(0, 0, -1, 0);
                    light_intensity = TraceRayRadiance(scene, *rt, hit, L, distance(in.WorldPosition, hit.V.WorldPosition) / max(view.ViewportSize.y, 1.0f), ctx.Media, hit.Valid ? distance(in.WorldPosition, hit.V.WorldPosition) : INFINITY);
                    if (ctx.Sample) {
                        ctx.Sample->DirectionPdf = float4(L, 1.0f / pdf);
                        ctx.Sample->IncidentDistance = float4(light_intensity, hit.Valid ? distance(in.WorldPosition, hit.V.WorldPosition) : 1e10f);
                    }
                }
                light_intensity /= float(Mode == PbrMode::Reuse ? 1u : ray_count) * pdf;
                point_to_light = L;
            }
            const float3 H = EnableDiffuseTrans ? safeNormalize(L + v, n) : normalize(L + v);
            const float NdotL = clampedDot(n, L);
            const float NdotH = clampedDot(n, H);
            const float VdotH = clampedDot(v, H);

            float3 dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(VdotH));
            float3 metal_fresnel = F_Schlick(base_color.rgb, float3(1.0f), abs(VdotH));

            if (area) {
                dielectric_fresnel = area_dielectric_fresnel;
                metal_fresnel = area_metal_fresnel;
            }
            const float area_diffuse = area ? AreaIntegral(area_lut, emitter.Shape, L, emitter.Rectangle, area_diffuse_data) : 0.0f;
            float3 l_diffuse = area ? light_intensity * area_diffuse * base_color.rgb : light_intensity * NdotL * BRDF_lambertian(base_color.rgb);
            if (diffuse_transmission_factor > 0.0f) {
                l_diffuse *= (1.0f - diffuse_transmission_factor);
                if (area || dot(n, L) < 0.0f) {
                    float3 l_diffuse_btdf = area ? light_intensity * AreaDiffuse(area_lut, emitter, -n, v) * diffuse_transmission_color : light_intensity * clampedDot(-n, L) * BRDF_lambertian(diffuse_transmission_color);
                    const float3 l_mirror = normalize(L + 2.0f * n * dot(-L, n));
                    if (!area) dielectric_fresnel = F_Schlick(f0_dielectric * specular_weight, f90_dielectric, abs(clampedDot(v, normalize(l_mirror + v))));
                    l_diffuse_btdf = applyVolumeAttenuation(l_diffuse_btdf, world_thickness, attenuation_color, attenuation_distance);
                    l_diffuse += l_diffuse_btdf * diffuse_transmission_factor;
                }
            }
            if (indirect && !TraceDiffuse) l_diffuse = float3(0);
            if (transmission_factor > 0.0f) {
                const float3 transmission_ray = getVolumeTransmissionRay(n, v, world_thickness, material.Ior);
                const float3 transmit_l = safeNormalize(point_to_light - transmission_ray, L);
                float3 l_transmit = area ? light_intensity * base_color.rgb * AreaTransmission(area_lut, emitter, n, v, perceptual_roughness, material.Ior, material.Volume.Shape == ThicknessShape::Slab ? -world_thickness : world_thickness) : light_intensity * getPunctualRadianceTransmission(n, v, transmit_l, alpha_roughness, base_color.rgb, material.Ior);
                l_transmit = applyVolumeAttenuation(l_transmit, length(transmission_ray), attenuation_color, attenuation_distance);
                l_diffuse = mix(l_diffuse, l_transmit, transmission_factor);
            }
            if constexpr (Mode == PbrMode::GlassDirect) l_diffuse = float3(0);
            // EEVEE light_eval_single shares visibility across the material closures.
            if constexpr (Mode == PbrMode::GlassSurface) l_diffuse *= 1.0f - ctx.GlassFraction;
            const float3 l_specular = area ? light_intensity * AreaIntegral(area_lut, emitter.Shape, L, emitter.Rectangle, area_specular_data) : (Mode == PbrMode::Reuse && reflection_weight == 0.0f) ? float3(0) : light_intensity * NdotL * (has_anisotropy ? BRDF_specularGGXAnisotropy(alpha_roughness, anisotropy_strength, n, v, L, H, anisotropic_t, anisotropic_b) : Mode == PbrMode::Raster ? BRDF_specularGGX(alpha_roughness, NdotL, NdotV, NdotH) :
                                                                                                                                                                                                                                                                                             float3(V_GGX(NdotL, NdotV, alpha_roughness) * RayGgxDistribution(n, H, alpha_roughness)));

            float3 diffuse_only = l_diffuse * (1.0f - metallic) * (1.0f - dielectric_fresnel);
            if (has_iridescence) diffuse_only = mix(diffuse_only, l_diffuse * (1.0f - metallic) * (1.0f - iridescence_fresnel_dielectric), iridescence_factor);
            float3 l_metal_brdf = metal_fresnel * l_specular;
            float3 l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);
            if (has_iridescence) {
                l_metal_brdf = mix(l_metal_brdf, l_specular * iridescence_fresnel_metallic, iridescence_factor);
                l_dielectric_brdf = mix(l_dielectric_brdf, mix(l_diffuse, l_specular, iridescence_fresnel_dielectric), iridescence_factor);
            }
            float3 l_color = mix(l_dielectric_brdf, l_metal_brdf, metallic);
            if (has_sheen) {
                const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
                const float l_albedo_sheen_scaling = area ? max(1.0f - max_sheen * sheen_lut_ndotv, 0.0f) : min(
                    1.0f - max_sheen * sheen_lut_ndotv,
                    1.0f - max_sheen * albedoSheenScalingLUT(scene, NdotL, sheen_roughness)
                );
                diffuse_only *= l_albedo_sheen_scaling;
                // EEVEE evaluates sheen/fuzz as a diffuse closure; retain the glTF sheen albedo LUT for its weight.
                const float3 sheen = area ? light_intensity * area_diffuse * sheen_color * sheen_lut_ndotv : light_intensity * NdotL * BRDF_specularSheen(sheen_color, sheen_roughness, NdotL, NdotV, NdotH);
                l_color = sheen + l_color * l_albedo_sheen_scaling;
            }
            if (has_clearcoat) {
                const float NdotL_cc = clampedDot(n_cc, L);
                const float NdotH_cc = clampedDot(n_cc, H);
                // Clearcoat Fresnel uses the microfacet half-angle instead of glTF-Sample-Viewer's constant view-angle approximation.
                const float3 F_cc = area ? eevee_opaque::fresnel(float3(f0_ior), float3(1), eevee_glass::lut2(area_lut, float2(cc_perceptual_roughness, sqrt(saturate(1.0f - NdotV_cc))), 1u).xy) : F_Schlick(float3(f0_ior), float3(1.0f), abs(VdotH));
                const float3 l_clearcoat = area ? light_intensity * F_cc * AreaGgx(area_lut, emitter, n_cc, v, cc_perceptual_roughness) : light_intensity * NdotL_cc * F_cc * (Mode == PbrMode::Raster ? BRDF_specularGGX(cc_alpha_roughness, NdotL_cc, NdotV_cc, NdotH_cc) : float3(V_GGX(NdotL_cc, NdotV_cc, cc_alpha_roughness) * RayGgxDistribution(n_cc, H, cc_alpha_roughness)));
                diffuse_only *= (1.0f - clearcoat_factor * F_cc);
                l_color = l_color * (1.0f - clearcoat_factor * F_cc) + clearcoat_factor * l_clearcoat;
            }
            if constexpr (indirect) l_color = diffuse_only + (l_color - diffuse_only) * reflection_weight;
            if constexpr (Mode == PbrMode::Reuse) ctx.Reuse->Accumulate(l_color * view.Exposure, reuse_pdf);
            else if (indirect) ray_color += l_color;
            else direct_color += l_color;
        }
    }

    if constexpr (Mode == PbrMode::Reuse) return float4(0);

    float ao = 1.0f;
    float3 indirect_color(0), cc_fresnel_ibl(0);
    if (Mode == PbrMode::Direct && ctx.OpaqueClosure) {
        if (material.OcclusionTexture.Slot != InvalidSlot) ao = ctx.SampleTexture(material.OcclusionTexture).r;
        const auto opaque = eevee_opaque::prepare(rt->Lut, n, v, perceptual_roughness, base_color.rgb, metallic, f0_dielectric, specular_weight, 1.0f + material.OcclusionStrength * (ao - 1.0f));
        *ctx.OpaqueClosure = opaque;
        if (!view.RayFastGI && (!TraceDiffuse || reflection_weight < 1.0f)) {
            const float3 diffuse_ibl = getDiffuseLight(scene, n) * opaque.Diffuse;
            if (!TraceDiffuse) indirect_color += diffuse_ibl;
            if (reflection_weight < 1.0f) {
                const float3 raster_indirect = diffuse_ibl + getIBLRadianceGGX(scene, n, v, perceptual_roughness) * opaque.Reflection;
                indirect_color += (raster_indirect - diffuse_ibl) * (1.0f - reflection_weight);
            }
        }
    } else {
        float3 f_diffuse = getDiffuseLight(scene, n) * base_color.rgb;
        if (diffuse_transmission_factor > 0.0f) {
            float3 f_diffuse_transmission = getDiffuseLight(scene, -n) * diffuse_transmission_color;
            f_diffuse_transmission = applyVolumeAttenuation(f_diffuse_transmission, world_thickness, attenuation_color, attenuation_distance);
            f_diffuse = mix(f_diffuse, f_diffuse_transmission, diffuse_transmission_factor);
        }
        if (transmission_factor > 0.0f) {
            const bool real = !TransmissionPrepass && view.UseRealTransmission != 0u && view.TransmissionFramebufferSamplerSlot != InvalidSlot;
            float3 f_transmission = getVolumeRefraction(scene, n, v, in.WorldPosition, world_thickness, perceptual_roughness, material.Ior, material.Dispersion, real) * base_color.rgb;
            f_transmission = applyVolumeAttenuation(f_transmission, world_thickness, attenuation_color, attenuation_distance);
            f_diffuse = mix(f_diffuse, f_transmission, transmission_factor);
        }
        const float3 f_specular_dielectric = has_anisotropy ? getIBLRadianceAnisotropy(scene, n, v, perceptual_roughness, anisotropy_strength, anisotropic_b) : getIBLRadianceGGX(scene, n, v, perceptual_roughness);
        const float3 f_specular_metal = f_specular_dielectric;

        const float2 ibl_brdf_f_ab = scene.SampleTex(view.Ibl.BrdfLutSamplerSlot, clamp(float2(NdotV, perceptual_roughness), float2(0.0f), float2(1.0f))).rg;
        const float3 f_metal_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, base_color.rgb, 1.0f);
        float3 f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

        const float3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(ibl_brdf_f_ab, NdotV, perceptual_roughness, f0_dielectric, specular_weight);
        float3 f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

        if (has_iridescence) {
            f_metal_brdf_ibl = mix(f_metal_brdf_ibl, f_specular_metal * iridescence_fresnel_metallic, iridescence_factor);
            f_dielectric_brdf_ibl = mix(f_dielectric_brdf_ibl, mix(f_diffuse, f_specular_dielectric, iridescence_fresnel_dielectric), iridescence_factor);
        }
        indirect_color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, metallic);
        if (has_sheen) {
            const float3 f_sheen = getIBLRadianceCharlie(scene, n, v, sheen_roughness, sheen_color);
            const float max_sheen = max(sheen_color.r, max(sheen_color.g, sheen_color.b));
            const float albedo_sheen_scaling = 1.0f - max_sheen * sheen_lut_ndotv;
            indirect_color = f_sheen + indirect_color * albedo_sheen_scaling;
        }
        if (material.OcclusionTexture.Slot != InvalidSlot) {
            ao = ctx.SampleTexture(material.OcclusionTexture).r;
            indirect_color *= (1.0f + material.OcclusionStrength * (ao - 1.0f));
        }
        if (has_clearcoat) {
            // Clearcoat IBL Fresnel uses the split-sum BRDF LUT with multi-scattering compensation instead of glTF-Sample-Viewer's Schlick term.
            cc_fresnel_ibl = getIBLGGXFresnel(scene, n_cc, v, cc_perceptual_roughness, float3(f0_ior), 1.0f);
            const float3 f_clearcoat_ibl = cc_fresnel_ibl * getIBLRadianceGGX(scene, n_cc, v, cc_perceptual_roughness);
            indirect_color = indirect_color * (1.0f - clearcoat_factor * cc_fresnel_ibl) + clearcoat_factor * f_clearcoat_ibl;
        }

        if constexpr (Mode != PbrMode::Raster && Mode != PbrMode::Secondary && Mode != PbrMode::GlassSurface) {
            const float3 raster_indirect = indirect_color;
            indirect_color = float3(0);
            if constexpr (Mode == PbrMode::Direct) {
                float3 diffuse_ibl = (1.0f - metallic) * f_diffuse * (1.0f - f_dielectric_fresnel_ibl);
                if (has_sheen) diffuse_ibl *= 1.0f - max(sheen_color.r, max(sheen_color.g, sheen_color.b)) * sheen_lut_ndotv;
                if (has_clearcoat) diffuse_ibl *= 1.0f - clearcoat_factor * cc_fresnel_ibl;
                if (!TraceDiffuse) indirect_color += diffuse_ibl;
                indirect_color += (raster_indirect - diffuse_ibl) * (1.0f - reflection_weight);
            }
        }
    }
    if constexpr (Mode == PbrMode::GlassSurface) indirect_color *= 1.0f - ctx.GlassFraction;
    float3 color = direct_color + indirect_color;
    if (traced_radiance) *traced_radiance = ray_color * (ctx.Secondary ? 1.0f : view.Exposure);
    else color += ray_color;
    float3 emissive = float3(material.EmissiveFactor) * material.EmissiveStrength;
    if (material.EmissiveTexture.Slot != InvalidSlot) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
    if (has_clearcoat) emissive *= (1.0f - clearcoat_factor * cc_fresnel_ibl);
    color += emissive;

    if (material.AlphaMode == MaterialAlphaMode::Mask) {
        if constexpr (Fragment)
            if (!ctx.Secondary && base_color.a < material.AlphaCutoff) discard_fragment();
        base_color.a = 1.0f;
    }

    if (view.DebugChannel != DebugChannel::None) {
        float3 dbg = float3(0.0f);
        switch (view.DebugChannel) {
            case DebugChannel::UvCoords0: dbg = float3(in.TexCoord0, 0.0f); break;
            case DebugChannel::UvCoords1: dbg = float3(in.TexCoord1, 0.0f); break;
            case DebugChannel::NormalTexture: dbg = (normal_info.ntex + 1.0f) * 0.5f; break;
            case DebugChannel::NormalGeometry: dbg = (normal_info.ng + 1.0f) * 0.5f; break;
            case DebugChannel::NormalShading: dbg = (n + 1.0f) * 0.5f; break;
            case DebugChannel::Tangent: dbg = (normal_info.t + 1.0f) * 0.5f; break;
            case DebugChannel::Bitangent: dbg = (normal_info.b + 1.0f) * 0.5f; break;
            case DebugChannel::TangentW: dbg = float3((in.WorldTangent.w + 1.0f) * 0.5f); break;
            case DebugChannel::Alpha: dbg = float3(base_color.a); break;
            case DebugChannel::Occlusion: dbg = float3(ao); break;
            case DebugChannel::Emissive: dbg = linearTosRGB(emissive); break;
            case DebugChannel::BaseColor: dbg = linearTosRGB(base_color.rgb); break;
            case DebugChannel::Metallic: dbg = float3(metallic); break;
            case DebugChannel::Roughness: dbg = float3(perceptual_roughness); break;
            case DebugChannel::ClearcoatFactor: dbg = float3(clearcoat_factor); break;
            case DebugChannel::ClearcoatRoughness: dbg = float3(cc_perceptual_roughness); break;
            case DebugChannel::ClearcoatNormal: dbg = (n_cc + 1.0f) * 0.5f; break;
            case DebugChannel::SheenColor: dbg = sheen_color; break;
            case DebugChannel::SheenRoughness: dbg = float3(sheen_roughness); break;
            case DebugChannel::SpecularFactor: dbg = float3(specular_weight); break;
            case DebugChannel::SpecularColor: dbg = specular_color; break;
            case DebugChannel::TransmissionFactor: dbg = float3(transmission_factor); break;
            case DebugChannel::VolumeThickness: {
                const float denom = material.Volume.ThicknessFactor * in.WorldScale;
                dbg = denom > 0.0f ? float3(world_thickness / denom) : float3(0.0f);
            } break;
            case DebugChannel::DiffuseTransmissionFactor: dbg = linearTosRGB(float3(diffuse_transmission_factor)); break;
            case DebugChannel::DiffuseTransmissionColor: dbg = linearTosRGB(diffuse_transmission_color); break;
            case DebugChannel::IridescenceFactor: dbg = float3(iridescence_factor); break;
            // Iridescence thickness normalizes to the reference implementation's 1200 nm range.
            case DebugChannel::IridescenceThickness: dbg = float3(iridescence_thickness / 1200.0f); break;
            case DebugChannel::AnisotropyStrength: dbg = float3(anisotropy_strength); break;
            case DebugChannel::AnisotropyDirection: dbg = float3((anisotropy_dir + 1.0f) * 0.5f, 0.0f); break;
        }
        return float4(dbg, base_color.a);
    }

    if (!TransmissionPrepass && !ctx.Secondary) color *= view.Exposure;

    const uint overlay_flags = in.FaceOverlayFlags & 3u;
    if (overlay_flags != 0u) {
        constant ViewportThemeColors &colors = scene.Theme.Colors;
        const bool is_edit_face = view.InteractionMode == InteractionMode::Edit && view.EditElement == Element::Face;
        const float4 selected = is_edit_face ? float4(colors.FaceSelected) : float4(colors.FaceSelectedIncidental);
        const float3 overlay = (overlay_flags & 2u) != 0u ? mix(selected.rgb, float4(colors.ElementActive).rgb, 0.5f) : selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    return float4(color, base_color.a);
}

fragment TransparencyStore PbrTransparentFragment(
    MeshletVertexVaryings meshlet_in [[stage_in]],
    TransparencyValues values [[imageblock_data]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5), function_constant(EnableAlphaBlend)]],
    device const RayTracingInstance *instances [[buffer(6), function_constant(EnableAlphaBlend)]],
    device const uint *triangles [[buffer(7), function_constant(EnableAlphaBlend)]],
    device const float *environment [[buffer(8), function_constant(EnableAlphaBlend)]],
    texture2d_array<float> light_lut [[texture(15)]]
) {
    const MeshVaryings in = FromMeshletVertexVaryings(meshlet_in);
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, in};
    ctx.LightLut = light_lut;
    const uint topology = NonTriangleTopology ? meshlet_in.Topology : uint(MeshPrimitiveTopology::Triangle);
    float4 shaded;
    if (EnableAlphaBlend) {
        const RayScene rays{acceleration, instances, triangles, environment, light_lut};
        const RayMediumState media = RayCameraState(rays);
        ctx.Media = &media;
        ctx.Opaque = ctx.RayTracing = true;
        ctx.GeometricNormal = normalize(cross(dfdy(in.WorldPosition), dfdx(in.WorldPosition)));
        shaded = ShadePbr<PbrMode::Secondary, true>(in, topology, meshlet_in.PointCoord, scene, view, ctx, &rays);
        shaded.rgb *= RayMediumAttenuation(scene, media, distance(float3(view.CameraPosition), in.WorldPosition));
    } else shaded = ShadePbr(in, topology, meshlet_in.PointCoord, scene, view, ctx);
    return StoreTransparency(values, shaded, in.Position.z);
}

fragment float4 PbrVisibilityFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    texture2d_array<float> light_lut [[texture(15)]]
) {
    DecodedSurface decoded = DecodeVisibility(quad.Position.xy, visibility, bindless, view, theme, workspace, pc);
    if (!decoded.Valid) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, decoded.V};
    ctx.LightLut = light_lut;
    ctx.SetSurface(decoded);
    return ShadePbr(
        decoded.V, NonTriangleTopology ? decoded.Topology : uint(MeshPrimitiveTopology::Triangle),
        decoded.PointCoord, scene, view, ctx
    );
}

// Keep the material context local to one hit. Inlining the layer lighting into
// the coverage walk produced intermittent HDR channel errors on blended surfaces.
__attribute__((noinline)) float3 ShadeCoverageLayer(const thread Scene &scene, const thread RayScene &rt, const thread DecodedSurface &surface, float3 direction, float footprint, float coverage, bool allow_feedback) {
    device const PBRMaterial &m = scene.Materials(scene.View.MaterialSlot)[surface.V.MaterialIndex];
    PbrContext ctx{scene, surface.V};
    ctx.SetSurface(surface);
    ctx.Secondary = ctx.Opaque = true;
    ctx.Outgoing = -direction;
    ctx.Footprint = footprint / max(min(length(surface.PositionDx), length(surface.PositionDy)), 1e-6f);

    float3 color(0);
    if (EnableTransmission && !m.Unlit && m.Transmission.Factor > 0.0f) {
        // Secondary deferred glass uses the environment probe, as EEVEE's probe fallback does.
        const auto cl = PrepareGlass(ctx, rt);
        float3 origin;
        const float4 reflected = eevee_glass::generate(cl, 0u, float2(0), origin);
        const float4 refracted = eevee_glass::generate(cl, 1u, float2(0), origin);
        color = cl.Reflection * getSpecularSample(scene, reflected.xyz, cl.Roughness * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u)).rgb;
        if (dot(refracted.xyz, refracted.xyz) > 0.0f) color += cl.Transmission * getSpecularSample(scene, refracted.xyz, cl.roughness(1u) * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u)).rgb;
        ctx.GlassFraction = cl.Fraction;
        if (cl.Fraction < 1.0f) color += ShadePbr<PbrMode::GlassSurface>(surface.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, scene.View, ctx, &rt).rgb;
        else color += ShadePbr<PbrMode::GlassDirect>(surface.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, scene.View, ctx, &rt).rgb;
    } else if (coverage > 0.0f) {
        const bool cached = rt.HistoryValid && allow_feedback && RayCameraState(rt).Count == 0u && dot(surface.GeometricNormal, -direction) * dot(surface.GeometricNormal, float3(scene.View.CameraPosition) - surface.V.WorldPosition) > 0.0f && m.AlphaMode != MaterialAlphaMode::Blend &&
            RayFeedback(rt.Feedback, rt.HistoryGeometry, rt.HistoryNormal, rt.HistoryMaterial, rt.Motion, rt.PreviousViewProj, rt.PreviousClipToWorld, rt.ApplyMotion,
                surface.V.WorldPosition, surface.GeometricNormal, surface.InstanceSlot, surface.V.MaterialIndex, color);
        if (cached) color /= max(scene.View.Exposure, 1e-8f);
        else color = ShadePbr<PbrMode::Secondary>(surface.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, scene.View, ctx, &rt).rgb;
    }
    return color;
}
// Secondary lighting uses native shading at the Metal geometry hit.
inline float3 TraceRayRadiance(const thread Scene &scene, const thread RayScene &rt, DecodedSurface surface, float3 direction, float footprint, const thread RayMediumState *initial_media, float initial_distance, thread float *camera_hole) {
    float3 result(0), throughput = initial_media ? RayMediumAttenuation(scene, *initial_media, initial_distance) : float3(1);
    float uncovered = 1.0f;
    if (camera_hole) *camera_hole = 0.0f;
    for (uint depth = 0u; depth < 64u; ++depth) {
        if (!surface.Valid) {
            if (camera_hole) *camera_hole = uncovered;
            else result += throughput * getSpecularSample(scene, direction, 0.0f).rgb;
            break;
        }
        device const PBRMaterial &m = scene.Materials(scene.View.MaterialSlot)[surface.V.MaterialIndex];
        const float coverage = EnableAlphaBlend && m.AlphaMode == MaterialAlphaMode::Blend ? clamp(RaySurfaceAlpha(scene, surface, m), 0.0f, 1.0f) : 1.0f;
        const float3 color = ShadeCoverageLayer(scene, rt, surface, direction, footprint, coverage, camera_hole == nullptr);
        result += throughput * coverage * color;
        if (coverage >= 1.0f) break;
        throughput *= 1.0f - coverage;
        uncovered *= 1.0f - coverage;
        const float3 position = surface.V.WorldPosition;
        surface = TraceSurface(scene, rt, RayOrigin(position, surface.GeometricNormal, direction), direction);
    }
    return result;
}
