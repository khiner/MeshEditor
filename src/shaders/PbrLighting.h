#pragma once
#include "EeveeDenoise.h"
#include "LightCullingLib.h"
#include "PbrMaterial.h"
#include "tonemapping.metal"

// Radiance arriving from one direction.
// An area light carries its emitter, and `L` points to its center.
struct PbrLight {
    float3 L, ToLight, Radiance;
    bool Area;
    const thread AreaEmitter &Emitter;
};

// Light-independent area-light terms, which EEVEE prepares once per surface.
struct PbrAreaTerms {
    eevee_ltc::LTCData Diffuse, Back, Specular, Coat;
};
__attribute__((always_inline)) inline PbrAreaTerms AreaTerms(constant SceneViewUBO &view, const thread PbrClosure &cl, texture2d_array<float> lut) {
    PbrAreaTerms area{};
    if (EnableAreaLights && EnableSceneLights && view.UseSceneLightsRender && view.LightCount) {
        area.Diffuse = eevee_ltc::diffuse(cl.N, dot(cl.N, cl.V) < 0.0f ? -cl.V : cl.V);
        area.Specular = eevee_ltc::ggx(lut, cl.N, cl.V, cl.Roughness);
        if (cl.DiffuseTransmission > 0.0f) area.Back = eevee_ltc::diffuse(-cl.N, dot(cl.N, cl.V) > 0.0f ? -cl.V : cl.V);
        if (cl.HasClearcoat) area.Coat = eevee_ltc::ggx(lut, cl.ClearcoatN, cl.V, cl.ClearcoatRoughness);
    }
    return area;
}

// One light's radiance toward the view, from the diffuse group (diffuse reflection, diffuse transmission and specular transmission) and from the glossy group (the GGX, sheen and coat lobes).
struct PbrLightColor {
    float3 Diffuse, Glossy;
};
// Evaluates the diffuse group only when `diffuse` is true.
__attribute__((always_inline)) inline PbrLightColor ReflectLight(const thread PbrClosure &cl, const thread PbrLight &light, const thread PbrAreaTerms &area, texture2d_array<float> lut, bool diffuse) {
    const float3 n = cl.N, v = cl.V, L = light.L;
    float3 l_diffuse(0), l_glossy;
    if (light.Area) {
        const thread AreaEmitter &emitter = light.Emitter;
        const float diffuse_area = DiffuseArea(lut, emitter, area.Diffuse);
        if (diffuse) {
            l_diffuse = cl.DiffuseWeight * diffuse_area;
            if (cl.DiffuseTransmission > 0.0f) l_diffuse += cl.DiffuseTransmissionWeight * DiffuseArea(lut, emitter, area.Back);
            if (cl.Transmission > 0.0f) {
                device const PBRMaterial &material = *cl.Material;
                l_diffuse += cl.TransmissionWeight * AreaTransmission(lut, emitter, n, v, cl.Roughness, material.Ior, material.Volume.Shape == ThicknessShape::Slab ? -cl.Thickness : cl.Thickness);
            }
        }
        l_glossy = cl.SpecularAlbedo * GgxArea(lut, emitter, area.Specular);
        if (cl.HasSheen) l_glossy += cl.SheenWeight * cl.SheenAlbedo * diffuse_area;
        if (cl.HasClearcoat) l_glossy += cl.CoatAlbedo * GgxArea(lut, emitter, area.Coat);
    } else {
        const float3 h = EnableDiffuseTrans ? safeNormalize(L + v, n) : normalize(L + v);
        const float NdotL = clampedDot(n, L), NdotH = clampedDot(n, h), VdotH = clampedDot(v, h);
        const float alpha = GgxAlpha(cl.Roughness);
        if (diffuse) {
            l_diffuse = cl.DiffuseWeight * DiffuseValue(n, L);
            if (cl.DiffuseTransmission > 0.0f) l_diffuse += cl.DiffuseTransmissionWeight * DiffuseValue(-n, L);
            if (cl.Transmission > 0.0f) l_diffuse += cl.TransmissionWeight * TransmissionValue(n, v, safeNormalize(light.ToLight - cl.TransmissionRay, L), alpha, cl.Material->Ior);
        }
        const float specular = cl.HasAnisotropy ? GgxAnisotropicValue(alpha, cl.AnisotropyStrength, n, v, L, h, cl.AnisotropicT, cl.AnisotropicB, NdotL) : GgxValue(alpha, NdotL, cl.NdotV, NdotH);
        l_glossy = specular * F_Schlick(cl.SpecularF0, cl.SpecularF90, VdotH);
        if (cl.HasSheen) l_glossy += cl.SheenWeight * SheenValue(cl.SheenAlpha, NdotL, cl.NdotV, NdotH);
        if (cl.HasClearcoat) {
            const float3 n_cc = cl.ClearcoatN;
            l_glossy += GgxValue(GgxAlpha(cl.ClearcoatRoughness), clampedDot(n_cc, L), cl.ClearcoatNdotV, clampedDot(n_cc, h)) * F_Schlick(float3(cl.CoatF0), float3(cl.CoatF90), VdotH);
        }
    }
    return {l_diffuse * light.Radiance, l_glossy * light.Radiance};
}

// Image-based light toward the view in the same two groups.
// Occlusion applies to the layers beneath the coat.
struct PbrImageLight {
    float3 Diffuse, Glossy;
    float Occlusion;
};
__attribute__((always_inline)) inline PbrImageLight ImageLight(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, const thread PbrClosure &cl) {
    device const PBRMaterial &material = *cl.Material;
    const float3 n = cl.N, v = cl.V;
    const float3 irradiance = DiffuseEnvironment(scene, n);
    float3 diffuse = cl.DiffuseWeight * irradiance;
    if (cl.DiffuseTransmission > 0.0f) diffuse += cl.DiffuseTransmissionWeight * DiffuseEnvironment(scene, -n);
    if (cl.Transmission > 0.0f) {
        const bool real = !TransmissionPrepass && view.UseRealTransmission != 0u && view.TransmissionFramebufferSamplerSlot != InvalidSlot;
        diffuse += cl.TransmissionWeight * getVolumeRefraction(scene, n, v, ctx.In.WorldPosition, cl.Thickness, cl.Roughness, material.Ior, material.Dispersion, real);
    }
    float3 glossy = cl.SpecularAlbedo * (cl.HasAnisotropy ? getIBLRadianceAnisotropy(scene, n, v, cl.Roughness, cl.AnisotropyStrength, cl.AnisotropicB) : GgxEnvironment(scene, n, v, cl.Roughness));
    if (cl.HasSheen) glossy += cl.SheenWeight * cl.SheenAlbedo * irradiance;
    const PbrOcclusion occlusion = ctx.Occlusion(material);
    diffuse *= occlusion.Scale;
    glossy *= occlusion.Scale;
    if (cl.HasClearcoat) glossy += cl.CoatAlbedo * GgxEnvironment(scene, cl.ClearcoatN, v, cl.ClearcoatRoughness);
    return {diffuse, glossy, occlusion.Value};
}

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

__attribute__((always_inline)) inline uint RaySeed(const thread Scene &scene, const thread PbrContext &ctx) {
    const float3 position = ctx.In.WorldPosition;
    return RayHash(as_type<uint>(position.x) ^ RayHash(as_type<uint>(position.y)) ^ RayHash(as_type<uint>(position.z))) ^ RayHash(scene.View.RayFrame + ctx.SampleOffset);
}

// Scales the light by its visibility along shadow rays.
// Returns false for a light that cannot reach the closure.
// The primary pass of a camera pixel shares `primary_noise` across lights when each light takes one shadow ray.
__attribute__((always_inline)) inline bool ShadowLight(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt, const thread PbrClosure &cl, LightRecord record, thread PbrLight &light,
    thread uint &rng, bool primary, float2 primary_noise) {
    constant SceneViewUBO &view = scene.View;
    const thread MeshVaryings &in = ctx.In;
    // Reject zero-contribution punctual lights before tracing their visibility.
    if (all(light.Radiance == float3(0)) ||
        (!light.Area && dot(cl.N, light.L) <= 0.0f && (!cl.HasClearcoat || dot(cl.ClearcoatN, light.L) <= 0.0f) &&
         cl.DiffuseTransmission == 0.0f && cl.Transmission == 0.0f)) return false;
    float2 noise{};
    if (light.Area) {
        // EEVEE combines a per-pixel blue-noise tile with a temporal sequence.
        const float2 rotation = light.Emitter.NoiseRotation;
        noise = ctx.Secondary ? float2(RayRandom(rng), RayRandom(rng)) : fract(((!primary || MultipleShadowSamples || EnableTransmission || EnableAlphaBlend) ? rt.Lut.read(uint2(in.Position.xy) % 64u, 0u).xy : primary_noise) + rotation + RaySampleOffset(view.RayFrame + ctx.SampleOffset));
    }
    if (MultipleShadowSamples && light.Area) {
        RayMediumState media{};
        if (ctx.Media) media = *ctx.Media;
        light.Radiance *= AreaShadowVisibility(scene, rt, in.WorldPosition, ctx.GeometricNormal, light.Emitter, noise, media);
    } else {
        float3 shadow_vector = light.ToLight;
        if (light.Area) {
            const float2 sample = eevee_ltc::sample_shape(noise, light.Emitter.Rectangle);
            shadow_vector += light.Emitter.X * sample.x + light.Emitter.Y * sample.y;
        }
        const float3 shadow_direction = light.Area ? normalize(shadow_vector) : light.L;
        const float3 origin = RayOrigin(in.WorldPosition, ctx.GeometricNormal, shadow_direction);
        const float maximum = record.Type == LightType::Directional ? INFINITY : length(shadow_vector) - distance(origin, in.WorldPosition);
        if (EnableTransmission || EnableAlphaBlend) {
            RayMediumState media{};
            if (ctx.Media) media = *ctx.Media;
            light.Radiance *= EnableTransmission ? RayTransmittance(scene, rt.Geometry(), origin, shadow_direction, maximum, view.RayTransmissionShadows, media) : RayTransmittance<false>(scene, rt.Geometry(), origin, shadow_direction, maximum, false);
        }
#ifdef FILTER_AREA_SHADOWS
        else {
            const bool lit = !RayOccluded(scene, rt.Geometry(), origin, shadow_direction, maximum, OpaqueShadows, CullShadowBackfaces);
            if (primary && light.Area)
                light.Radiance *= ctx.ShadowNeighbors.visibility(lit);
            else if (!lit) light.Radiance = float3(0);
        }
#else
        else if (RayOccluded(scene, rt.Geometry(), origin, shadow_direction, maximum, OpaqueShadows, CullShadowBackfaces)) light.Radiance = float3(0);
#endif
    }
    return true;
}

__attribute__((always_inline)) inline uint SceneLightCount(constant SceneViewUBO &view) { return EnableSceneLights && view.UseSceneLightsRender != 0u ? view.LightCount : 0u; }

// Calls `shade(record, light)` for each scene light that reaches the closure.
template<typename Shade>
__attribute__((always_inline)) inline void ForEachLight(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, Shade shade) {
    const float3 position = ctx.In.WorldPosition;
    const uint light_count = SceneLightCount(view);
    device const LightRecord *lights = scene.Lights(view.LightSlot);
    auto light_iterator = TiledLightIterator::make(scene, position, light_count, TiledLights);
    for (uint i = light_iterator.next(TiledLights); i < light_count; i = light_iterator.next(TiledLights)) {
        float3 L{}, point_to_light{}, light_intensity{};
        AreaEmitter emitter{};
        const LightRecord record = lights[i];
        const bool area = EnableAreaLights && IsAreaLight(record.Type);
        if (area) {
            if (!PrepareAreaEmitter(scene, record, position, emitter, i)) continue;
            point_to_light = emitter.Center;
            L = normalize(point_to_light);
            light_intensity = emitter.Radiance;
        } else light_intensity = getLightIntensity(scene, record, position, L, point_to_light);
        PbrLight light{L, point_to_light, light_intensity, area, emitter};
        shade(record, light);
    }
}

// The final color of a lit surface: emission, alpha masking, debug channels, exposure and face overlays.
template<bool Fragment>
__attribute__((always_inline)) inline float4 PbrSurfaceColor(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, const thread PbrClosure &cl, float3 color, float ao) {
    const thread MeshVaryings &in = ctx.In;
    device const PBRMaterial &material = *cl.Material;
    float4 base_color = cl.BaseColor;
    float3 emissive = float3(material.EmissiveFactor) * material.EmissiveStrength;
    if (ctx.Samples(material.EmissiveTexture, PbrInput::Emissive)) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
    if (ctx.Computes(PbrInput::Emissive)) emissive = ctx.Computed3(PbrInput::Emissive) * material.EmissiveStrength;
    if (cl.HasClearcoat) emissive *= cl.CoatThroughput;
    color += emissive;

    if (material.AlphaMode == MaterialAlphaMode::Mask) {
        if constexpr (Fragment)
            if (!ctx.Secondary && base_color.a < material.AlphaCutoff) discard_fragment();
        base_color.a = 1.0f;
    }

    if (view.DebugChannel != DebugChannel::None) {
        const NormalInfo normal_info = cl.Normal;
        float3 dbg = float3(0.0f);
        switch (view.DebugChannel) {
            case DebugChannel::UvCoords0: dbg = float3(in.TexCoord0, 0.0f); break;
            case DebugChannel::UvCoords1: dbg = float3(in.TexCoord1, 0.0f); break;
            case DebugChannel::NormalTexture: dbg = (normal_info.ntex + 1.0f) * 0.5f; break;
            case DebugChannel::NormalGeometry: dbg = (normal_info.ng + 1.0f) * 0.5f; break;
            case DebugChannel::NormalShading: dbg = (cl.N + 1.0f) * 0.5f; break;
            case DebugChannel::Tangent: dbg = (normal_info.t + 1.0f) * 0.5f; break;
            case DebugChannel::Bitangent: dbg = (normal_info.b + 1.0f) * 0.5f; break;
            case DebugChannel::TangentW: dbg = float3((in.WorldTangent.w + 1.0f) * 0.5f); break;
            case DebugChannel::Alpha: dbg = float3(base_color.a); break;
            case DebugChannel::Occlusion: dbg = float3(ao); break;
            case DebugChannel::Emissive: dbg = linearTosRGB(emissive); break;
            case DebugChannel::BaseColor: dbg = linearTosRGB(base_color.rgb); break;
            case DebugChannel::Metallic: dbg = float3(cl.Metallic); break;
            case DebugChannel::Roughness: dbg = float3(cl.Roughness); break;
            case DebugChannel::ClearcoatFactor: dbg = float3(cl.Clearcoat); break;
            case DebugChannel::ClearcoatRoughness: dbg = float3(cl.ClearcoatRoughness); break;
            case DebugChannel::ClearcoatNormal: dbg = (cl.ClearcoatN + 1.0f) * 0.5f; break;
            case DebugChannel::SheenColor: dbg = cl.SheenColor; break;
            case DebugChannel::SheenRoughness: dbg = float3(cl.SheenRoughness); break;
            case DebugChannel::SpecularFactor: dbg = float3(cl.SpecularWeight); break;
            case DebugChannel::SpecularColor: dbg = cl.SpecularColor; break;
            case DebugChannel::TransmissionFactor: dbg = float3(cl.Transmission); break;
            case DebugChannel::VolumeThickness: {
                const float denom = material.Volume.ThicknessFactor * in.WorldScale;
                dbg = denom > 0.0f ? float3(cl.Thickness / denom) : float3(0.0f);
            } break;
            case DebugChannel::DiffuseTransmissionFactor: dbg = linearTosRGB(float3(cl.DiffuseTransmission)); break;
            case DebugChannel::DiffuseTransmissionColor: dbg = linearTosRGB(cl.DiffuseTransmissionColor); break;
            case DebugChannel::IridescenceFactor: dbg = float3(cl.Iridescence); break;
            // Iridescence thickness normalizes to the reference implementation's 1200 nm range.
            case DebugChannel::IridescenceThickness: dbg = float3(cl.IridescenceThickness / 1200.0f); break;
            case DebugChannel::AnisotropyStrength: dbg = float3(cl.AnisotropyStrength); break;
            case DebugChannel::AnisotropyDirection: dbg = float3((cl.AnisotropyDirection + 1.0f) * 0.5f, 0.0f); break;
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

// Scene lights with ray-traced shadows and image-based lighting, for secondary hits and for surfaces off the deferred path.
template<bool Fragment = false>
inline float4 ShadeSecondary(const thread Scene &scene, const thread PbrContext &ctx, uint topology, float2 point_coord, const thread RayScene &rt) {
    float4 base_color, unlit;
    if (ShadeUnlit<Fragment>(scene, scene.View, ctx, topology, point_coord, base_color, unlit)) return unlit;
    const PbrClosure cl = EvaluatePbrMaterial<Fragment>(scene, scene.View, ctx, base_color, rt.Lut);
    const PbrAreaTerms area = AreaTerms(scene.View, cl, rt.Lut);
    uint rng = RaySeed(scene, ctx);
    float3 direct(0);
    ForEachLight(scene, scene.View, ctx, [&](LightRecord record, thread PbrLight &light) __attribute__((always_inline)) {
        if (!ShadowLight(scene, ctx, rt, cl, record, light, rng, false, float2(0))) return;
        const PbrLightColor l = ReflectLight(cl, light, area, rt.Lut, true);
        direct += l.Diffuse + l.Glossy;
    });
    const PbrImageLight image = ImageLight(scene, scene.View, ctx, cl);
    return PbrSurfaceColor<Fragment>(scene, scene.View, ctx, cl, direct + image.Diffuse + image.Glossy, image.Occlusion);
}
