// Fast math that keeps Inf and NaN semantics.
#pragma METAL fp math_mode(relaxed)
#pragma once
#include "PbrLighting.h"
#include "RayEnvironmentSampling.h"
#include "RayReuse.h"
#include "RayPatternHits.h"
#include "RayWork.h"
#include "gpu/RayLightingPushConstants.h"

// The extra deferred lobes a scene traces: the coat with reflections, diffuse transmission with diffuse light, and sheen with diffuse light or else with reflections.
// They share the rays of the reflection and diffuse lobes.
constant bool TraceCoat = EnableClearcoat && TraceReflections;
constant bool TraceDiffuseTransmission = EnableDiffuseTrans && TraceDiffuse;
constant bool TraceSheen = EnableSheen && (TraceReflections || TraceDiffuse);
constant bool TraceExtraLobes = TraceCoat || TraceDiffuseTransmission || TraceSheen;

struct RayReuseSample {
    float4 DirectionPdf;
    float4 IncidentDistance;
};

inline eevee_glass::Closure PrepareGlass(const thread PbrContext &ctx, const thread RayScene &rt) {
    device const PBRMaterial &m = ctx.S.Materials(ctx.S.View.MaterialSlot)[ctx.In.MaterialIndex];
    eevee_glass::Closure cl{};
    cl.N = ctx.GetNormalInfo<false>(m).n;
    cl.V = ctx.ViewDirection();
    cl.Position = ctx.In.WorldPosition;
    cl.Ior = max(m.Ior, 1.0f);
    const float2 mr = saturate(ctx.MetallicRoughness(m));
    cl.Roughness = mr.y;
    cl.Fraction = saturate(ctx.Transmission(m)) * (1.0f - mr.x);
    const float thickness = ctx.Thickness(m);
    cl.Thickness = m.Volume.Shape == ThicknessShape::Slab ? -thickness : thickness;
    float3 tint = float3(m.BaseColorFactor.rgb) * ctx.In.VertexColor.rgb;
    if (ctx.Samples(m.BaseColorTexture, PbrInput::BaseColor)) tint *= ctx.SampleTexture(m.BaseColorTexture).rgb;
    if (ctx.Computes(PbrInput::BaseColor)) tint = ctx.Computed3(PbrInput::BaseColor) * ctx.In.VertexColor.rgb;
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
    float4 Reflection, Transmission, GlassShape, Coat, DiffuseTransmission, Sheen;
    float2 CoatNormal;
};
inline RayClosureWeights GlassWeights(const thread eevee_glass::Closure &cl) {
    return {float4(cl.Reflection, cl.Roughness), float4(cl.Transmission, cl.roughness(1u)), float4(EncodeRayNormal(cl.N), cl.Roughness, cl.Thickness)};
}

// Glossy reflection of the scene lights by a surface whose glass closure carries transmission and diffuse light.
inline float4 ShadeGlassDirect(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt) {
    float4 base_color, unlit;
    if (ShadeUnlit<false>(scene, scene.View, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), base_color, unlit)) return unlit;
    const PbrClosure cl = EvaluatePbrMaterial<false>(scene, scene.View, ctx, base_color, rt.Lut);
    const PbrAreaTerms area = AreaTerms(scene.View, cl, rt.Lut);
    uint rng = RaySeed(scene, ctx);
    float3 direct(0);
    ForEachLight(scene, scene.View, ctx, [&](LightRecord record, thread PbrLight &light) __attribute__((always_inline)) {
        if (ShadowLight(scene, ctx, rt, cl, record, light, rng, false, float2(0))) direct += ReflectLight(cl, light, area, rt.Lut, false).Glossy;
    });
    return PbrSurfaceColor<false>(scene, scene.View, ctx, cl, direct, ctx.Occlusion(*cl.Material).Value);
}

// Scene lights and image-based lighting of the part of a surface outside its glass closure.
// EEVEE light_eval_single shares visibility across the material closures.
inline float4 ShadeGlassSurface(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt, float glass_fraction) {
    float4 base_color, unlit;
    if (ShadeUnlit<false>(scene, scene.View, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), base_color, unlit)) return unlit;
    const PbrClosure cl = EvaluatePbrMaterial<false>(scene, scene.View, ctx, base_color, rt.Lut);
    const PbrAreaTerms area = AreaTerms(scene.View, cl, rt.Lut);
    uint rng = RaySeed(scene, ctx);
    float3 direct(0);
    ForEachLight(scene, scene.View, ctx, [&](LightRecord record, thread PbrLight &light) __attribute__((always_inline)) {
        if (!ShadowLight(scene, ctx, rt, cl, record, light, rng, false, float2(0))) return;
        const PbrLightColor l = ReflectLight(cl, light, area, rt.Lut, true);
        direct += l.Diffuse * (1.0f - glass_fraction) + l.Glossy;
    });
    const PbrImageLight image = ImageLight(scene, scene.View, ctx, cl);
    return PbrSurfaceColor<false>(scene, scene.View, ctx, cl, direct + (image.Diffuse + image.Glossy) * (1.0f - glass_fraction), image.Occlusion);
}

// Keep the material context local to one hit. Inlining the layer lighting into
// the coverage walk produced intermittent HDR channel errors on blended surfaces.
__attribute__((noinline)) float3 ShadeCoverageLayer(const thread Scene &scene, const thread RayScene &rt, const thread DecodedSurface &surface, float3 direction, float footprint, float coverage, bool allow_feedback, uint pattern_word) {
    device const PBRMaterial &m = scene.Materials(scene.View.MaterialSlot)[surface.V.MaterialIndex];
    PbrContext ctx{scene, surface.V};
    ctx.SetSurface(surface, pattern_word);
    ctx.Secondary = ctx.Opaque = true;
    ctx.Outgoing = -direction;
    ctx.Footprint = HitFootprint(surface, footprint);

    float3 color(0);
    if (EnableTransmission && !m.Unlit && m.Transmission.Factor > 0.0f) {
        // Secondary deferred glass uses the environment probe, as EEVEE's probe fallback does.
        const auto cl = PrepareGlass(ctx, rt);
        float3 origin;
        const float4 reflected = eevee_glass::generate(cl, 0u, float2(0), origin);
        const float4 refracted = eevee_glass::generate(cl, 1u, float2(0), origin);
        color = cl.Reflection * getSpecularSample(scene, reflected.xyz, cl.Roughness * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u)).rgb;
        if (dot(refracted.xyz, refracted.xyz) > 0.0f) color += cl.Transmission * getSpecularSample(scene, refracted.xyz, cl.roughness(1u) * float(max(scene.View.Ibl.SpecularEnvMipCount, 1u) - 1u)).rgb;
        if (cl.Fraction < 1.0f) color += ShadeGlassSurface(scene, ctx, rt, cl.Fraction).rgb;
        else color += ShadeGlassDirect(scene, ctx, rt).rgb;
    } else if (coverage > 0.0f) {
        const bool cached = rt.HistoryValid && allow_feedback && RayCameraState(rt).Count == 0u && dot(surface.GeometricNormal, -direction) * dot(surface.GeometricNormal, float3(scene.View.CameraPosition) - surface.V.WorldPosition) > 0.0f && m.AlphaMode != MaterialAlphaMode::Blend &&
            RayFeedback(rt.Feedback, rt.HistoryGeometry, rt.HistoryNormal, rt.HistoryMaterial, rt.Motion, rt.PreviousViewProj, rt.PreviousClipToWorld, rt.ApplyMotion,
                surface.V.WorldPosition, surface.GeometricNormal, surface.InstanceSlot, surface.V.MaterialIndex, color);
        if (cached) color /= max(scene.View.Exposure, 1e-8f);
        else color = ShadeSecondary(scene, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), rt).rgb;
    }
    return color;
}
// Secondary lighting uses native shading at the Metal geometry hit.
// The first surface reads the inputs its pattern computed from `pattern_word`, and the surfaces behind it read their data inputs.
inline float3 TraceRayRadiance(const thread Scene &scene, const thread RayScene &rt, DecodedSurface surface, float3 direction, float footprint, const thread RayMediumState *initial_media, float initial_distance, thread float *camera_hole = nullptr, uint pattern_word = 0u) {
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
        const float3 color = ShadeCoverageLayer(scene, rt, surface, direction, footprint, coverage, camera_hole == nullptr, depth == 0u ? pattern_word : 0u);
        result += throughput * coverage * color;
        if (coverage >= 1.0f) break;
        throughput *= 1.0f - coverage;
        uncovered *= 1.0f - coverage;
        const float3 position = surface.V.WorldPosition;
        surface = TraceRayHit(scene, rt, RayOrigin(position, surface.GeometricNormal, direction), direction).Surface;
    }
    return result;
}

// Lists a ray sample whose first surface's material computes inputs, keeping the surface's hit in the sample's dead ray origin and the list entry.
// Each SIMD group that lists any reserves its entries with one atomic.
inline void DeferPatternHit(device uint *hits, device RayLightingSample &sample, uint queue_index, uint4 hit) {
    sample.OriginFootprint.xyz = as_type<float3>(hit.xyz);
    const uint rank = simd_prefix_exclusive_sum(1u), count = simd_sum(1u);
    uint first = 0u;
    if (simd_is_first()) first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(hits), count, memory_order_relaxed);
    const uint at = simd_broadcast_first(first) + rank;
    hits[PatternHitEntries + 2u * at] = queue_index;
    hits[PatternHitEntries + 2u * at + 1u] = hit.w;
}

// Selection probabilities of the lobes a traced ray samples, in proportion to their weighted albedos.
// The diffuse lobe transmits the `Transmitted` fraction of its samples.
struct RayLobeWeights {
    float Diffuse, Specular, Sheen, Coat, Transmitted;
};
__attribute__((always_inline)) inline RayLobeWeights SampleWeights(const thread PbrClosure &cl, float reflection_weight) {
    constexpr float3 Average(1.0f / 3.0f);
    const float reflected = TraceDiffuse ? dot(cl.DiffuseWeight, Average) : 0.0f;
    const float transmitted = TraceDiffuse ? dot(cl.DiffuseTransmissionWeight, Average) : 0.0f;
    const bool glossy = reflection_weight > 0.0f;
    const float specular = glossy ? dot(cl.SpecularAlbedo, Average) : 0.0f;
    const float sheen = glossy && cl.HasSheen ? dot(cl.SheenWeight, Average) * cl.SheenAlbedo : 0.0f;
    const float coat = glossy && cl.HasClearcoat ? cl.CoatAlbedo : 0.0f;
    const float diffuse = reflected + transmitted, total = diffuse + specular + sheen + coat;
    const float scale = total > 0.0f ? 1.0f / total : 0.0f;
    return {diffuse * scale, specular * scale, sheen * scale, coat * scale, diffuse > 0.0f ? transmitted / diffuse : 0.0f};
}
__attribute__((always_inline)) inline float SamplePdf(const thread PbrClosure &cl, RayLobeWeights weights, float3 L) {
    return (weights.Diffuse > 0.0f ? weights.Diffuse * DiffusePdf(cl.N, L, weights.Transmitted) : 0.0f) +
        (weights.Specular > 0.0f ? weights.Specular * GgxPdf(cl.N, cl.V, L, GgxAlpha(cl.Roughness)) : 0.0f) +
        (weights.Sheen > 0.0f ? weights.Sheen * SheenPdf(cl.N, L) : 0.0f) +
        (weights.Coat > 0.0f ? weights.Coat * GgxPdf(cl.ClearcoatN, cl.V, L, GgxAlpha(cl.ClearcoatRoughness)) : 0.0f);
}
// Samples a direction from the lobe that `component` selects, or returns zero for a sample the lobe rejects.
__attribute__((always_inline)) inline float3 SampleLobe(const thread PbrClosure &cl, RayLobeWeights weights, float2 u, float component) {
    if (component < weights.Diffuse) return DiffuseSample(cl.N, u, component < weights.Diffuse * weights.Transmitted);
    if (component < weights.Diffuse + weights.Sheen) return SheenSample(cl.N, u);
    const bool coat = component >= weights.Diffuse + weights.Sheen + weights.Specular;
    const float3 n = coat ? cl.ClearcoatN : cl.N;
    const auto sample = GgxSample(n, cl.V, GgxAlpha(coat ? cl.ClearcoatRoughness : cl.Roughness), u);
    return sample.pdf > 0.0f ? sample.direction : float3(0);
}
// Whether a sampled direction leaves the geometric surface on the side its lobe scatters to.
__attribute__((always_inline)) inline bool SampleFacing(const thread PbrContext &ctx, const thread PbrClosure &cl, float3 L) {
    const bool transmitted = TraceDiffuse && cl.DiffuseTransmission > 0.0f && dot(cl.N, L) < 0.0f;
    const float geometric_side = dot(ctx.GeometricNormal, L) * dot(ctx.GeometricNormal, cl.V);
    return (transmitted ? -geometric_side : geometric_side) > 0.0f;
}

// The roughness of the closure's smoothest specular lobe.
inline float TraceRoughness(const thread PbrClosure &cl) { return cl.HasClearcoat ? min(cl.Roughness, cl.ClearcoatRoughness) : cl.Roughness; }
// The weight of traced reflection on a lobe of `roughness`, which is zero when the scene traces no reflections.
inline float TracedReflection(float roughness, constant SceneViewUBO &view) { return TraceReflections ? RayReflectionWeight(roughness, view.RayMaxRoughness) : 0.0f; }

// Traces rays sampled from the closure and returns the radiance they reflect toward the camera.
// The first ray's hit goes to `hit_guide` and the last ray's sample to `sample`.
inline float3 ShadeIndirect(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt, uint sample_count, thread float &roughness, thread float4 &hit_guide, thread RayReuseSample &sample) {
    float4 base_color, unlit;
    if (ShadeUnlit<false>(scene, scene.View, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), base_color, unlit)) return float3(0);
    const PbrClosure cl = EvaluatePbrMaterial<false>(scene, scene.View, ctx, base_color, rt.Lut);
    constant SceneViewUBO &view = scene.View;
    const float3 position = ctx.In.WorldPosition;
    roughness = TraceRoughness(cl);
    const float reflection_weight = TracedReflection(roughness, view);
    const uint ray_count = (reflection_weight > 0.0f || TraceDiffuse) ? max(sample_count, 1u) : 0u;
    const RayLobeWeights weights = SampleWeights(cl, reflection_weight);
    const AreaEmitter no_emitter{};
    uint rng = RaySeed(scene, ctx);
    float3 ray_color(0);
    for (uint i = 0u; i < ray_count; ++i) {
        const float2 u = float2(RayRandom(rng), RayRandom(rng));
        const float component = RayRandom(rng);
        float3 L = SampleLobe(cl, weights, u, component);
        const float environment_weight = view.EnvIntensity > 0.0f && (weights.Diffuse > 0.0f || weights.Sheen > 0.0f || cl.Roughness > 0.15f) ? 0.5f : 0.0f;
        if (RayRandom(rng) < environment_weight) L = transpose(view.EnvRotation.Unpack()) * SampleRayEnvironment(rt.Environment, float2(RayRandom(rng), RayRandom(rng)));
        const float pdf = (1.0f - environment_weight) * SamplePdf(cl, weights, L) + environment_weight * RayEnvironmentPdf(rt.Environment, view.EnvRotation.Unpack() * L);
        if (pdf <= 0.0f || !SampleFacing(ctx, cl, L)) continue;
        const PbrLightColor l = ReflectLight(cl, PbrLight{L, L, float3(1.0f / (float(ray_count) * pdf)), false, no_emitter}, PbrAreaTerms{}, rt.Lut, TraceDiffuse);
        const float3 throughput = l.Diffuse + l.Glossy * reflection_weight;
        const DecodedSurface hit = TraceRayHit(scene, rt, RayOrigin(position, ctx.GeometricNormal, L), L).Surface;
        if (i == 0u) hit_guide = hit.Valid ? float4(EncodeRayNormal(hit.GeometricNormal), distance(position, hit.V.WorldPosition), float(hit.V.MaterialIndex)) : float4(0, 0, -1, 0);
        const float3 light_intensity = TraceRayRadiance(scene, rt, hit, L, distance(position, hit.V.WorldPosition) / max(view.ViewportSize.y, 1.0f), ctx.Media, hit.Valid ? distance(position, hit.V.WorldPosition) : INFINITY);
        sample.DirectionPdf = float4(L, 1.0f / pdf);
        sample.IncidentDistance = float4(light_intensity, hit.Valid ? distance(position, hit.V.WorldPosition) : 1e10f);
        ray_color += throughput * light_intensity;
    }
    return ray_color * view.Exposure;
}

// Accumulates the light of the neighborhood's traced samples, reflected by this pixel's closure, into the spatial reuse.
inline void ShadeReuse(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt, thread RaySpatialReuse &reuse) {
    float4 base_color, unlit;
    if (ShadeUnlit<false>(scene, scene.View, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), base_color, unlit)) return;
    const PbrClosure cl = EvaluatePbrMaterial<false>(scene, scene.View, ctx, base_color, rt.Lut);
    constant SceneViewUBO &view = scene.View;
    const float reflection_weight = TracedReflection(TraceRoughness(cl), view);
    const uint ray_count = (reflection_weight > 0.0f || TraceDiffuse) ? reuse.Count : 0u;
    const RayLobeWeights weights = SampleWeights(cl, reflection_weight);
    const AreaEmitter no_emitter{};
    for (uint i = 0u; i < ray_count; ++i) {
        float3 L;
        if (!reuse.Sample(i, L)) continue;
        const float pdf = SamplePdf(cl, weights, L);
        if (pdf <= 0.0f || !SampleFacing(ctx, cl, L)) continue;
        const PbrLight light{L, L, reuse.Radiance / pdf, false, no_emitter};
        const PbrLightColor l = ReflectLight(cl, light, PbrAreaTerms{}, rt.Lut, TraceDiffuse);
        reuse.Accumulate((l.Diffuse + l.Glossy * reflection_weight) * view.Exposure, pdf);
    }
}

// Anisotropic reflection takes the per-sample path, since its lobe has no sample of its own.
inline bool DeferredOpaque(device const PBRMaterial &m) {
    return !m.Unlit && m.AlphaMode != MaterialAlphaMode::Blend && m.Transmission.Factor == 0.0f && !(EnableAnisotropy && m.Anisotropy.Strength != 0.0f);
}
// The closure of a surface on the deferred ray path: the weights of its GGX reflection and its diffuse reflection, which the traced light of each lobe multiplies.
struct DeferredClosure {
    float3 N, Reflection, Diffuse;
    float Roughness;
};
// Returns the direction of a ray of a diffuse or GGX lobe about `n`, and its inverse probability.
// GGX samples skip the grazing part of the cap.
inline float4 DeferredSample(float3 n, float roughness, float3 v, bool diffuse, float2 noise) {
    if (diffuse) {
        const float3 direction = DiffuseSample(n, noise, false);
        return float4(direction, M_PI_F / max(dot(n, direction), 1e-8f));
    }
    const auto sample = GgxSample(n, v, GgxAlpha(roughness), float2(1.0f - noise.x * 0.92f, noise.y));
    return float4(sample.direction, sample.pdf > 0.0f ? 1.0f / sample.pdf : 0.0f);
}

// A lobe of a ray that several lobes share: its normal, roughness, sample kind and selection weight.
enum class DeferredKind : uint { Ggx, Diffuse, Sheen };
struct DeferredSharedLobe {
    float3 N;
    float Roughness;
    DeferredKind Kind;
    float Weight;
};
inline float DeferredSharedPdf(DeferredSharedLobe lobe, float3 v, float3 direction) {
    return lobe.Kind == DeferredKind::Ggx ? GgxPdf(lobe.N, v, direction, GgxAlpha(lobe.Roughness)) :
        lobe.Kind == DeferredKind::Diffuse ? DiffusePdf(lobe.N, direction, 0.0f) : SheenPdf(lobe.N, direction);
}
// The lobes one deferred ray carries at a pixel, each with its color weight, whose average selects it, and the integral of its value.
// A lobe that faces away from the shading normal `N` scatters through the surface.
struct DeferredRay {
    float3 N;
    DeferredSharedLobe Lobes[3];
    float3 Weights[3];
    float Albedo[3];
    uint Count;
    float Total; // The sum of the lobes' selection weights
    void Add(float3 n, float roughness, DeferredKind kind, float3 weight, float albedo) {
        if (all(weight <= float3(0))) return;
        Lobes[Count] = {n, roughness, kind, max(dot(weight, float3(1.0f / 3.0f)), 1e-6f)};
        Total += Lobes[Count].Weight;
        Weights[Count] = weight;
        Albedo[Count++] = albedo;
    }
    // The sum of the lobes' weighted values for a ray direction, each value normalized to integrate to one.
    // A GGX or diffuse lobe's normalized value is its probability.
    float3 Value(float3 v, float3 direction) const {
        float3 value(0);
        for (uint i = 0u; i < Count; ++i) {
            const DeferredSharedLobe lobe = Lobes[i];
            const float3 n = lobe.N;
            const float NdotL = dot(n, direction);
            const float f = lobe.Kind != DeferredKind::Sheen ? DeferredSharedPdf(lobe, v, direction) :
                NdotL > 0.0f ? SheenValue(SheenAlpha(lobe.Roughness), NdotL, saturate(dot(n, v)), saturate(dot(n, normalize(direction + v)))) / Albedo[i] : 0.0f;
            value += Weights[i] * f;
        }
        return value;
    }
};
// Samples the mixture of the ray's lobes in proportion to their weights, with `select` choosing the lobe.
// Returns the direction and the inverse probability of the mixture, and whether the chosen lobe scatters through the surface.
inline float4 DeferredSharedSample(const thread DeferredRay &ray, float3 v, float2 noise, float select, thread bool &transmit) {
    const thread DeferredSharedLobe *lobes = ray.Lobes;
    const uint count = ray.Count;
    float target = select * ray.Total;
    uint chosen = count - 1u;
    for (uint i = 0u; i + 1u < count; ++i) {
        if (target < lobes[i].Weight) {
            chosen = i;
            break;
        }
        target -= lobes[i].Weight;
    }
    const DeferredSharedLobe lobe = lobes[chosen];
    const float3 direction = lobe.Kind == DeferredKind::Ggx ? DeferredSample(lobe.N, lobe.Roughness, v, false, noise).xyz :
        lobe.Kind == DeferredKind::Diffuse ? DiffuseSample(lobe.N, noise, false) : SheenSample(lobe.N, noise);
    float pdf = 0.0f;
    for (uint i = 0u; i < count; ++i) pdf += lobes[i].Weight / ray.Total * DeferredSharedPdf(lobes[i], v, direction);
    transmit = dot(lobe.N, ray.N) < 0.0f;
    return float4(direction, pdf > 0.0f ? 1.0f / pdf : 0.0f);
}
// The lobes ray `lobe` of a deferred pixel carries, from the closure the direct pass stored: the ray's weight in `weight`, the shading normal, the base roughness in the guide, and the extra lobes' weights.
// The reflection ray carries the base reflection, the coat, and sheen without traced diffuse light, and the diffuse ray carries the diffuse reflection, diffuse transmission, and sheen with traced diffuse light.
// Fast GI stores the whole base reflection, of which the reflection ray carries the part its roughness allows.
inline DeferredRay DeferredLobes(constant SceneViewUBO &view, uint lobe, uint2 pixel, float4 guide, float3 v, texture2d<float, access::read> weight, texture2d<float, access::read> normal,
    texture2d<float, access::read> coat_weight, texture2d<float, access::read> coat_normal, texture2d<float, access::read> diffuse_transmission_weight, texture2d<float, access::read> sheen_weight) {
    DeferredRay ray{.N = DecodeRayNormal(normal.read(pixel).xy)};
    const float3 n = ray.N;
    const bool coated = TraceCoat && lobe == 0u, transmits = TraceDiffuseTransmission && lobe == 1u, sheen_lobe = TraceSheen && lobe == (TraceDiffuse ? 1u : 0u);
    const float4 coat = coated ? coat_weight.read(pixel) : float4(0), sheen = sheen_lobe ? sheen_weight.read(pixel) : float4(0);
    const float3 transmitted = transmits ? diffuse_transmission_weight.read(pixel).rgb : float3(0);
    const float roughness = RayGuideBaseRoughness(guide);
    const float traced = lobe == 0u && view.RayFastGI ? RayFastGiTraced(roughness, view.RayReflections, view.RayMaxRoughness) : 1.0f;
    ray.Add(n, roughness, lobe ? DeferredKind::Diffuse : DeferredKind::Ggx, max(weight.read(pixel).rgb - coat.rgb - sheen.rgb - transmitted, 0.0f) * traced, 1.0f);
    if (coated) ray.Add(DecodeRayNormal(coat_normal.read(pixel).xy), coat.a, DeferredKind::Ggx, coat.rgb, 1.0f);
    if (transmits) ray.Add(-n, 1.0f, DeferredKind::Diffuse, transmitted, 1.0f);
    if (sheen_lobe) ray.Add(n, sheen.a, DeferredKind::Sheen, sheen.rgb, SheenAlbedo(sheen.a, saturate(dot(n, v))));
    return ray;
}

// Direct scene light with ray-traced shadows on a camera pixel's surface.
// A deferred surface writes its closure to `opaque`, which carries its traced indirect light, and the rest takes image-based lighting.
// A scene that traces extra lobes also writes the weights of the coat, diffuse transmission and sheen lobes the reflection and diffuse rays carry to `extra`, with the coat's normal.
inline float4 ShadeDirect(const thread Scene &scene, const thread PbrContext &ctx, const thread RayScene &rt, thread DeferredClosure *opaque, thread RayClosureWeights *extra, thread float &roughness) {
    float4 base_color, unlit;
    if (ShadeUnlit<false>(scene, scene.View, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), base_color, unlit)) return unlit;
    const PbrClosure cl = EvaluatePbrMaterial<false>(scene, scene.View, ctx, base_color, rt.Lut);
    constant SceneViewUBO &view = scene.View;
    device const PBRMaterial &material = *cl.Material;
    roughness = TraceRoughness(cl);
    const float reflection_weight = TracedReflection(roughness, view);
    const PbrAreaTerms area = AreaTerms(view, cl, rt.Lut);
    float2 shadow_blue_noise{};
    if (!MultipleShadowSamples && !EnableTransmission && !EnableAlphaBlend && EnableAreaLights && SceneLightCount(view) && !ctx.Secondary) shadow_blue_noise = rt.Lut.read(uint2(ctx.In.Position.xy) % 64u, 0u).xy;
    uint rng = RaySeed(scene, ctx);
    float3 direct(0);
    ForEachLight(scene, scene.View, ctx, [&](LightRecord record, thread PbrLight &light) __attribute__((always_inline)) {
        if (!ShadowLight(scene, ctx, rt, cl, record, light, rng, true, shadow_blue_noise)) return;
        const PbrLightColor l = ReflectLight(cl, light, area, rt.Lut, true);
        direct += l.Diffuse + l.Glossy;
    });

    float ao = 1.0f;
    float3 indirect_color(0);
    if (opaque) {
        const PbrOcclusion surface_occlusion = ctx.Occlusion(material);
        ao = surface_occlusion.Value;
        const float occlusion = surface_occlusion.Scale;
        // Each GGX lobe traces the part of its light that its own roughness allows.
        const float base_weight = EnableClearcoat && TraceReflections ? RayReflectionWeight(cl.Roughness, view.RayMaxRoughness) : reflection_weight;
        const float coat_weight = cl.HasClearcoat && TraceReflections ? RayReflectionWeight(cl.ClearcoatRoughness, view.RayMaxRoughness) : 0.0f;
        // Sheen traces with diffuse light, or else with a reflection ray traced for the other lobes, and Fast GI gives it diffuse light.
        // Untraced sheen takes the diffuse lobe's light, as its prefiltered environment is the diffuse one.
        const bool trace_sheen = TraceSheen && cl.HasSheen && (TraceDiffuse || (!view.RayFastGI && (base_weight > 0.0f || coat_weight > 0.0f)));
        const float3 sheen = cl.SheenWeight * cl.SheenAlbedo * occlusion;
        *opaque = {cl.N, cl.SpecularAlbedo * occlusion, cl.HasSheen && !trace_sheen ? cl.DiffuseWeight * occlusion + sheen : cl.DiffuseWeight * occlusion, cl.Roughness};
        if (trace_sheen) extra->Sheen = float4(sheen, cl.SheenRoughness);
        if (!view.RayFastGI && (!TraceDiffuse || base_weight < 1.0f)) {
            const float3 diffuse_ibl = DiffuseEnvironment(scene, cl.N) * opaque->Diffuse;
            if (!TraceDiffuse) indirect_color += diffuse_ibl;
            if (base_weight < 1.0f) {
                const float3 raster_indirect = diffuse_ibl + GgxEnvironment(scene, cl.N, cl.V, cl.Roughness) * opaque->Reflection;
                indirect_color += (raster_indirect - diffuse_ibl) * (1.0f - base_weight);
            }
        }
        if (cl.HasClearcoat) {
            if (coat_weight < 1.0f) indirect_color += GgxEnvironment(scene, cl.ClearcoatN, cl.V, cl.ClearcoatRoughness) * cl.CoatAlbedo * (1.0f - coat_weight);
            if (TraceCoat) {
                extra->Coat = float4(float3(cl.CoatAlbedo * coat_weight), cl.ClearcoatRoughness);
                extra->CoatNormal = EncodeRayNormal(cl.ClearcoatN);
            }
        }
        if (cl.DiffuseTransmission > 0.0f) {
            const float3 transmitted = cl.DiffuseTransmissionWeight * occlusion;
            if (TraceDiffuse) extra->DiffuseTransmission = float4(transmitted, 1.0f);
            else indirect_color += DiffuseEnvironment(scene, -cl.N) * transmitted;
        }
    } else {
        const PbrImageLight image = ImageLight(scene, scene.View, ctx, cl);
        ao = image.Occlusion;
        if (!TraceDiffuse) indirect_color += image.Diffuse;
        indirect_color += image.Glossy * (1.0f - reflection_weight);
    }
    return PbrSurfaceColor<false>(scene, scene.View, ctx, cl, direct + indirect_color, ao);
}

// Writes the guide of a glass or blended surface, which leaves the opaque ray path, and returns true for one.
__attribute__((always_inline)) inline bool RayTransparentGuide(const thread PbrContext &ctx, const thread DecodedSurface &decoded, device const PBRMaterial &material, thread bool &blended, thread float4 &guide) {
    blended = EnableTransmission && EnableAlphaBlend && material.AlphaMode == MaterialAlphaMode::Blend;
    if (blended || (EnableTransmission && !material.Unlit && material.Transmission.Factor > 0.0f)) {
        // Glass transport is independent of the opaque reflection/GI controls.
        const float roughness = ctx.MetallicRoughness(material).y;
        guide = float4(EncodeRayNormal(decoded.GeometricNormal), blended ? 0.0f : 2.0f + clamp(roughness, 0.0f, 1.0f), blended ? -1.0f - float(decoded.V.MaterialIndex) : float(decoded.V.MaterialIndex));
        return true;
    }
    return false;
}

// Shades a camera pixel's direct light and writes its guide and the weights of its traced closures.
// With Prepare, only glass surfaces and points and lines are shaded.
template<bool Prepare = false>
inline float4 ShadeRayPixel(const thread Scene &scene, const thread RayScene &rays, const thread DecodedSurface &decoded,
    constant RayLightingPushConstants &pc, thread float4 &guide, thread RayClosureWeights &weights, uint pattern_word
#ifdef FILTER_AREA_SHADOWS
    , uint lane = 0u, uint visibility_id = 0u
#endif
) {
    PbrContext ctx{scene, decoded.V};
    ctx.SampleOffset = pc.SampleOffset;
    ctx.SetSurface(decoded, pattern_word);
    ctx.RayTracing = true;
    const RayMediumState camera_media = RayCameraState(rays);
    ctx.Media = &camera_media;
    if (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle)) {
        guide = float4(0, 0, -1, 0);
        ctx.GeometricNormal = ShadingWorldNormal(decoded.V);
        const float4 color = ShadeSecondary(scene, ctx, decoded.Topology, decoded.PointCoord, rays);
        const float3 attenuation = RayMediumAttenuation(scene, camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition));
        return float4(color.rgb * attenuation, color.a);
    }
    device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[decoded.V.MaterialIndex];
    bool blended;
    if (RayTransparentGuide(ctx, decoded, material, blended, guide)) {
        if (Prepare && !blended) weights = GlassWeights(PrepareGlass(ctx, rays));
        return float4(0, 0, 0, 1);
    }
    if constexpr (Prepare) return float4(0);
#ifdef FILTER_AREA_SHADOWS
    ctx.ShadowNeighbors = ShadowNeighborhood::make(visibility_id, uint2(decoded.V.Position.xy), lane);
#endif
    const bool deferred = DeferredOpaque(material);
    DeferredClosure opaque{};
    float roughness = -1.0f;
    const float4 base = ShadeDirect(scene, ctx, rays, deferred ? &opaque : nullptr, TraceExtraLobes && deferred ? &weights : nullptr, roughness);
    // The reflection ray carries the coat, and sheen without traced diffuse light.
    // A surface whose reflection ray carries one of them stays active whatever its base roughness.
    const float3 shared_sheen = TraceDiffuse ? float3(0) : weights.Sheen.rgb;
    const bool shared = (TraceCoat && any(weights.Coat.rgb != float3(0))) || (TraceSheen && any(shared_sheen != float3(0)));
    if (!TraceDiffuse && !(scene.View.RayFastGI && deferred) && !shared && TracedReflection(roughness, scene.View) == 0.0f) roughness = -1.0f;
    guide = float4(EncodeRayNormal(decoded.GeometricNormal), roughness, float(decoded.V.MaterialIndex));
    if (deferred && roughness >= 0.0f) {
        // With a coat, the roughness that keeps the surface active is the smoother of the coat's and the base's.
        const float base_roughness = EnableClearcoat ? opaque.Roughness : roughness;
        guide = float4(EncodeRayNormal(opaque.N), (shared ? 6.0f : 4.0f) + base_roughness, float(decoded.V.MaterialIndex));
        weights.Reflection = float4(opaque.Reflection * (scene.View.RayFastGI ? 1.0f : TracedReflection(base_roughness, scene.View)), base_roughness);
        weights.Transmission = float4((TraceDiffuse || scene.View.RayFastGI) ? opaque.Diffuse : float3(0), 1.0f);
        if (TraceExtraLobes) {
            // Each ray's weight is the sum of the lobes it carries, with the roughness of its smoothest lobe.
            const float reflection_roughness = any(weights.Reflection.rgb != float3(0)) ? base_roughness : 1.0f;
            const float coat_roughness = any(weights.Coat.rgb != float3(0)) ? weights.Coat.a : 1.0f;
            weights.Reflection = float4(weights.Reflection.rgb + weights.Coat.rgb + shared_sheen, min(reflection_roughness, coat_roughness));
            weights.Transmission.rgb += weights.DiffuseTransmission.rgb + (TraceDiffuse ? weights.Sheen.rgb : float3(0));
        }
    }
    const float3 attenuation = RayMediumAttenuation(scene, camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition));
    return float4(base.rgb * attenuation, base.a);
}

// Traces the indirect light of a camera pixel whose material is off the deferred path, and writes its guide.
// A blended surface returns the light through its coverage layers and their uncovered fraction in `camera_hole`.
inline float3 TraceRayPixel(const thread Scene &scene, const thread RayScene &rays, const thread DecodedSurface &decoded,
    constant RayLightingPushConstants &pc, thread float4 &guide, thread float4 &hit_guide, thread float &camera_hole, thread RayReuseSample &sample, uint pattern_word) {
    PbrContext ctx{scene, decoded.V};
    ctx.SampleOffset = pc.SampleOffset;
    ctx.SetSurface(decoded, pattern_word);
    ctx.RayTracing = true;
    const RayMediumState camera_media = RayCameraState(rays);
    ctx.Media = &camera_media;
    if (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle)) {
        guide = float4(0, 0, -1, 0);
        return float3(0);
    }
    device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[decoded.V.MaterialIndex];
    bool blended;
    if (RayTransparentGuide(ctx, decoded, material, blended, guide)) {
        if (!blended) return float3(0);
        return scene.View.Exposure * TraceRayRadiance(scene, rays, decoded, -ctx.ViewDirection(), max(length(decoded.PositionDx), length(decoded.PositionDy)), &camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition), &camera_hole, pattern_word);
    }
    float roughness = -1.0f;
    const float3 radiance = ShadeIndirect(scene, ctx, rays, pc.SingleSample ? 1u : scene.View.RaySamples, roughness, hit_guide, sample);
    if (!TraceDiffuse && !(scene.View.RayFastGI && DeferredOpaque(material)) && TracedReflection(roughness, scene.View) == 0.0f) roughness = -1.0f;
    guide = float4(EncodeRayNormal(decoded.GeometricNormal), roughness, float(decoded.V.MaterialIndex));
    const float3 attenuation = RayMediumAttenuation(scene, camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition));
    sample.IncidentDistance.rgb *= attenuation;
    return radiance * attenuation;
}

// The scene and ray tracing arguments of every lit ray kernel.
#define RAY_KERNEL_ARGS SCENE_ARGS, instance_acceleration_structure acceleration [[buffer(5)]], device const RayTracingInstance *instances [[buffer(6)]], device const uint *triangles [[buffer(7)]], device const float *environment [[buffer(8)]], texture2d_array<float> lut [[texture(15)]]
// The history and motion arguments of a ray kernel that reads cached light.
#define RAY_HISTORY_ARGS texture2d<float, access::read> feedback [[texture(16)]], texture2d<float, access::read> history_geometry [[texture(17)]], texture2d<float, access::read> history_normal [[texture(18)]], texture2d<uint, access::read> history_material [[texture(19)]], device const RayMotionTransform *motion [[buffer(10)]]

kernel void PbrOpaqueLayer(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<uint, access::read> primary_visibility [[texture(1)]],
    depth2d<float, access::read> depth [[texture(20)]],
    texture2d<float, access::write> radiance [[texture(3)]],
    texture2d<float, access::write> geometry [[texture(9)]],
    texture2d<float, access::write> normal [[texture(10)]],
    texture2d<uint, access::write> identity [[texture(11)]],
    RAY_KERNEL_ARGS,
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pixel.x >= visibility.get_width() || pixel.y >= visibility.get_height()) return;
    identity.write(uint4(0), pixel);
    geometry.write(float4(0), pixel);
    const uint2 primary_pixel = min(uint2((float2(pixel) + 0.5f) * float2(primary_visibility.get_width(), primary_visibility.get_height()) / float2(visibility.get_width(), visibility.get_height())), uint2(primary_visibility.get_width(), primary_visibility.get_height()) - 1u);
    if (visibility.read(pixel).r == primary_visibility.read(primary_pixel).r) return;
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    if (!decoded.Valid || (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle))) return;
    // A surface whose material computes inputs is not cached, so rays that hit it shade it from its computed inputs.
    if (EnablePatterns && BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[decoded.V.MaterialIndex].Program.Index != InvalidOffset) return;
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    PbrContext ctx{scene, decoded.V};
    ctx.SetSurface(decoded, 0u);
    ctx.Secondary = ctx.Opaque = true;
    ctx.Outgoing = view.ScreenPixelScale < 0.0f ? transpose(view.ViewRotation.Unpack()) * float3(0, 0, 1) : normalize(float3(view.CameraPosition) - decoded.V.WorldPosition);
    ctx.Footprint = float(visibility.get_width()) / float(primary_visibility.get_width());
    radiance.write(ShadeSecondary(scene, ctx, uint(MeshPrimitiveTopology::Triangle), float2(0), rays), pixel);
    geometry.write(float4(depth.read(pixel), as_type<float>(decoded.InstanceSlot + 1u), 0, 0), pixel);
    normal.write(float4(EncodeRayNormal(decoded.GeometricNormal), 0, 0), pixel);
    const auto resolved = ResolveVisibilityId(visibility.read(pixel).r, bindless, view, theme, workspace, pc.Visibility);
    identity.write(uint4(decoded.V.MaterialIndex + 1u, MeshletCoarse(resolved.Meshlet) ? 0u : resolved.Triangle + 1u, 0, 0), pixel);
}

kernel void PbrRayDirect(
#ifdef FILTER_AREA_SHADOWS
    uint lane [[thread_index_in_simdgroup]],
#endif
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::write> base_image [[texture(1)]],
    texture2d<float, access::write> radiance_image [[texture(2)]],
    texture2d<float, access::write> guide_image [[texture(3)]],
    texture2d<uint, access::write> history_material [[texture(4)]],
    texture2d<float, access::write> reflection_weight [[texture(10)]],
    texture2d<float, access::write> transmission_weight [[texture(11)]],
    texture2d<float, access::write> closure_normal [[texture(16)]],
    texture2d<float, access::write> glass_shape [[texture(21)]],
    texture2d<float, access::write> coat_weight [[texture(5), function_constant(TraceCoat)]],
    texture2d<float, access::write> coat_normal [[texture(6), function_constant(TraceCoat)]],
    texture2d<float, access::write> diffuse_transmission_weight [[texture(7), function_constant(TraceDiffuseTransmission)]],
    texture2d<float, access::write> sheen_weight [[texture(8), function_constant(TraceSheen)]],
    texture2d<float, access::write> history_geometry [[texture(17)]],
    texture2d<float, access::write> history_normal [[texture(18)]],
    depth2d<float, access::read> history_depth [[texture(20)]],
    RAY_KERNEL_ARGS,
    device uint *work [[buffer(9)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    // Inside a medium, prepare glass once and compact the opaque primary pixels.
    const bool compact = EnableTransmission && RayCameraState(rays).Count != 0u;
    if (PreparePrimary && !compact) {
        if (all(pixel == uint2(0))) work[0] = RayTileCount(visibility.get_width(), visibility.get_height()) * 64u;
        return;
    }
    if (EnableTransmission && !PreparePrimary) {
        const uint index = pixel.x;
        if (index >= work[0]) return;
        if (compact) {
            pixel = RayWorkPixel(work, index, visibility.get_width());
        } else {
            pixel = RayTilePixel(index / 64u, uint2(index % 8u, (index % 64u) / 8u), visibility.get_width());
            if (pixel.x >= visibility.get_width() || pixel.y >= visibility.get_height()) return;
        }
    }
    if (!EnableTransmission && all(pixel == uint2(0))) {
        work[0] = work[4] = 0u;
        if (TraceDiffuse && pc.DiffuseTileOffset) {
            work[pc.DiffuseTileOffset] = 0u;
            work[pc.DiffuseTileOffset + 1u] = work[pc.DiffuseTileOffset + 2u] = 1u;
        }
    }
    radiance_image.write(float4(0), pixel);
    history_material.write(uint4(0), pixel);
    history_geometry.write(float4(0), pixel);
#ifdef FILTER_AREA_SHADOWS
    const uint visibility_id = visibility.read(pixel).r;
    const auto decoded = DecodeVisibilityId(visibility_id, float2(pixel) + 0.5f, bindless, view, theme, workspace, pc.Visibility);
#else
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
#endif
    if (!decoded.Valid) {
        guide_image.write(float4(0, 0, -1, 0), pixel);
        return;
    }
    const Scene scene{bindless, view, theme, workspace};
    if (PreparePrimary) {
        device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[decoded.V.MaterialIndex];
        const bool transparent = (EnableAlphaBlend && material.AlphaMode == MaterialAlphaMode::Blend) || (EnableTransmission && !material.Unlit && material.Transmission.Factor > 0.f);
        if (!transparent) {
            RayAppendPixel(work, pixel.y * visibility.get_width() + pixel.x, true);
            return;
        }
    }

    float4 guide{};
    RayClosureWeights weights{};
    const uint pattern_word = PixelPatternWord(view, decoded.Topology, pixel);
    const float4 base = PreparePrimary ?
        ShadeRayPixel<true>(scene, rays, decoded, pc, guide, weights, pattern_word) :
        ShadeRayPixel(scene, rays, decoded, pc, guide, weights, pattern_word
#ifdef FILTER_AREA_SHADOWS
            , lane, visibility_id
#endif
        );
    if (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle)) {
        base_image.write(base, pixel);
        guide_image.write(float4(0, 0, -1, 0), pixel);
        return;
    }
    if (PreparePrimary && RayGuideGlass(guide)) {
        reflection_weight.write(weights.Reflection, pixel);
        transmission_weight.write(weights.Transmission, pixel);
        glass_shape.write(weights.GlassShape, pixel);
    }
    if (guide.z >= 4.0f) {
        reflection_weight.write(weights.Reflection, pixel);
        transmission_weight.write(weights.Transmission, pixel);
        closure_normal.write(float4(guide.xy, 0, 0), pixel);
        if (TraceCoat) {
            coat_weight.write(weights.Coat, pixel);
            coat_normal.write(float4(weights.CoatNormal, 0, 0), pixel);
        }
        if (TraceDiffuseTransmission) diffuse_transmission_weight.write(weights.DiffuseTransmission, pixel);
        if (TraceSheen) sheen_weight.write(weights.Sheen, pixel);
    }
    guide.xy = EncodeRayNormal(decoded.GeometricNormal);

    if (guide.w >= 0.0f) base_image.write(base, pixel);
    guide_image.write(guide, pixel);
    history_material.write(uint4(guide.w < 0.0f ? 0u : uint(guide.w) + 1u), pixel);
    history_geometry.write(float4(history_depth.read(pixel), as_type<float>((decoded.InstanceSlot + 1u) | (RayCameraState(rays).Count ? 0x80000000u : 0u)), 0, 0), pixel);
    history_normal.write(float4(EncodeRayNormal(decoded.GeometricNormal), 0, 0), pixel);
}

kernel void PbrRayGlassWeights(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> guide_image [[texture(3)]],
    texture2d<float, access::write> reflection_weight [[texture(10)]],
    texture2d<float, access::write> transmission_weight [[texture(11)]],
    RAY_KERNEL_ARGS,
    texture2d<float, access::write> glass_shape [[texture(21)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    if (RayCameraState(rays).Count != 0u || !RayGuideGlass(guide_image.read(pixel))) return;
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    const Scene scene{bindless, view, theme, workspace};

    PbrContext ctx{scene, decoded.V};
    ctx.SetSurface(decoded, PixelPatternWord(view, decoded.Topology, pixel));
    ctx.RayTracing = true;
    const auto weights = GlassWeights(PrepareGlass(ctx, rays));
    reflection_weight.write(weights.Reflection, pixel);
    transmission_weight.write(weights.Transmission, pixel);
    glass_shape.write(weights.GlassShape, pixel);
}

kernel void PbrRaySpecial(
    uint index [[thread_position_in_grid]],
    device const uint *work [[buffer(9)]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> guide_image [[texture(1)]],
    texture2d<float, access::write> radiance_image [[texture(2)]],
    texture2d<float, access::write> hit_image [[texture(4)]],
    texture2d<float, access::write> ray_samples [[texture(5)]],
    texture2d<float, access::write> incident [[texture(6)]],
    RAY_HISTORY_ARGS,
    RAY_KERNEL_ARGS,
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (index >= work[0]) return;
    const uint2 pixel = RayWorkPixel(work, index, visibility.get_width());
    if (RayGuideDeferred(guide_image.read(pixel))) return;
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut, feedback, history_geometry, history_normal, history_material,
        motion, pc.PreviousViewProj.Unpack(), pc.HistoryValid != 0u, pc.ApplyMotion != 0u, pc.PreviousClipToWorld.Unpack()};
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    float4 guide{};
    float4 hit_guide{0, 0, -2, 0};
    float hole = 0.0f;
    RayReuseSample sample{};
    const float3 radiance = TraceRayPixel(scene, rays, decoded, pc, guide, hit_guide, hole, sample, PixelPatternWord(view, decoded.Topology, pixel));
    if (guide.w >= 0.0f) {
        const uint step = view.RayStep, frame = view.RayFrame + pc.SampleOffset;
        const uint2 center = RayGridCenter(pixel / step, frame, step, uint2(visibility.get_width(), visibility.get_height()));
        if (all(pixel == center)) {
            ray_samples.write(sample.DirectionPdf, pixel / step);
            incident.write(sample.IncidentDistance, pixel / step);
        }
        hit_image.write(hit_guide, pixel);
    }
    radiance_image.write(float4(radiance, 1.0f + hole), pixel);
}

// The flags of a sample, with its light scale.
inline float RaySampleFlags(bool valid, bool glass, float scale) {
    return as_type<float>(RaySampleActive | (valid ? RaySampleValid : 0u) | (glass ? RaySampleGlass : 0u) | (uint(as_type<ushort>(half(min(scale, 65504.0f)))) << 16u));
}

kernel void PbrRayGenerate(
    uint index [[thread_position_in_grid]],
    device RayLightingSample *queue [[buffer(11)]],
    device const uint *work [[buffer(9)]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> guide_image [[texture(1)]],
    texture2d<float, access::write> direct_lighting [[texture(14)]],
    texture2d<float, access::read> transmission_weight [[texture(2), function_constant(TraceDiffuse)]],
    texture2d<float, access::read> reflection_weight [[texture(3)]],
    texture2d<float, access::read> closure_normal [[texture(21)]],
    texture2d<float, access::read> coat_weight [[texture(9), function_constant(TraceCoat)]],
    texture2d<float, access::read> coat_normal [[texture(10), function_constant(TraceCoat)]],
    texture2d<float, access::read> diffuse_transmission_weight [[texture(11), function_constant(TraceDiffuseTransmission)]],
    texture2d<float, access::read> sheen_weight [[texture(20), function_constant(TraceSheen)]],
    RAY_KERNEL_ARGS,
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (index >= work[0]) return;
    const uint2 pixel = RayWorkPixel(work, index, visibility.get_width());
    const float4 guide = guide_image.read(pixel);
    if (!RayGuideDeferred(guide)) return;
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    PbrContext ctx{scene, decoded.V};
    ctx.SetSurface(decoded, PixelPatternWord(view, decoded.Topology, pixel));
    ctx.RayTracing = true;
    const RayMediumState camera_media = RayCameraState(rays);
    ctx.Media = &camera_media;
    // Match the pre-split sample coordinates, including projection rounding.
    const float4 clip = view.ViewProj.Unpack() * float4(decoded.V.WorldPosition, 1);
    const uint2 sample_pixel = uint2(ndc_to_uv(clip.xy / clip.w) * view.ViewportSize);
    const float4 blue = lut.read(sample_pixel % 64u, 0u);
    const uint frame = view.RayFrame + pc.SampleOffset + pc.StagedSample;
    const float footprint = max(length(decoded.PositionDx), length(decoded.PositionDy));
    RayLightingSample samples[2]{};
    if (RayGuideGlass(guide)) {
        const auto cl = PrepareGlass(ctx, rays);
        if (pc.StagedSample == 0u) {
            ctx.Opaque = true;
            device const auto &material = scene.Materials(view.MaterialSlot)[decoded.V.MaterialIndex];
            float3 direct(0);
            if (cl.Fraction < 1.0f) direct = ShadeGlassSurface(scene, ctx, rays, cl.Fraction).rgb;
            else if ((EnableSceneLights && view.UseSceneLightsRender && view.LightCount) || any(float3(material.EmissiveFactor) > 0.0f)) direct = ShadeGlassDirect(scene, ctx, rays).rgb;
            direct_lighting.write(float4(direct, 1), pixel);
        }
        for (uint lobe = 0u; lobe < 2u; ++lobe) {
            const float2 noise = fract((lobe ? blue.zw : blue.xy) + RaySampleOffset(frame));
            float3 origin;
            const float4 ray = eevee_glass::generate(cl, lobe, noise, origin);
            const bool valid = ray.w > 0.0f && dot(ray.xyz, ray.xyz) > 0.5f;
            samples[lobe] = {ray, float4(RayOrigin(origin, cl.N, ray.xyz), footprint), float4(origin, RaySampleFlags(valid, true, 1.0f))};
        }
    } else {
        if (EnableTransmission && pc.StagedSample == 0u) direct_lighting.write(float4(0, 0, 0, 1), pixel);
        const float3 v = ctx.ViewDirection();
        uint rng = RayHash(pixel.x ^ RayHash(pixel.y ^ RayHash(frame)));
        for (uint lobe = 0u; lobe < (TraceDiffuse ? 2u : 1u); ++lobe) {
            const float2 noise = fract((lobe ? blue.zw : blue.xy) + RaySampleOffset(frame));
            const DeferredRay ray = DeferredLobes(view, lobe, pixel, guide, v, lobe ? transmission_weight : reflection_weight, closure_normal, coat_weight, coat_normal, diffuse_transmission_weight, sheen_weight);
            if (ray.Count == 0u) continue;
            // With extra lobes traced, a ray samples the mixture of its lobes, and otherwise its one lobe.
            bool transmit = false;
            const float4 direction = TraceExtraLobes ? DeferredSharedSample(ray, v, noise, RayRandom(rng), transmit) : DeferredSample(ray.N, ray.Lobes[0].Roughness, v, lobe != 0u, noise);
            // Sheen samples the hemisphere uniformly, so a ray carrying it scales its light by the ratio of the lobes' normalized value to the sample's probability.
            const float scale = TraceSheen && ray.Lobes[ray.Count - 1u].Kind == DeferredKind::Sheen ? dot(ray.Value(v, direction.xyz), float3(1.0f / 3.0f)) / ray.Total * direction.w : 1.0f;
            // A deferred ray leaves the geometric surface on the side its lobe scatters to.
            const bool valid = direction.w > 0.0f && dot(decoded.GeometricNormal, direction.xyz) * dot(decoded.GeometricNormal, v) * (transmit ? -1.0f : 1.0f) > 0.0f;
            samples[lobe] = {direction, float4(RayOrigin(decoded.V.WorldPosition, decoded.GeometricNormal, direction.xyz), footprint), float4(decoded.V.WorldPosition, RaySampleFlags(valid, false, scale))};
        }
    }
    const uint lobes = RayLightingLobeCount(EnableTransmission, TraceDiffuse);
    queue[lobes * index] = samples[0];
    if (lobes > 1u) queue[lobes * index + 1u] = samples[1];
}

kernel void PbrRayShade(
    uint2 thread_id [[thread_position_in_grid]],
    device const RayLightingSample *queue [[buffer(11)]],
    device const uint *work [[buffer(9)]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> guide_image [[texture(1)]],
    texture2d<float, access::read_write> radiance_image [[texture(2)]],
    texture2d<float, access::write> hit_image [[texture(4)]],
    texture2d<float, access::write> ray_samples [[texture(5)]],
    texture2d<float, access::write> incident [[texture(6)]],
    texture2d<float, access::read_write> transmission [[texture(7)]],
    texture2d<float, access::write> transmission_guide [[texture(8)]],
    texture2d<float, access::write> transmission_samples [[texture(12)]],
    texture2d<float, access::write> transmission_incident [[texture(13)]],
    texture2d<float, access::read> opaque_radiance [[texture(3), function_constant(EnableOpaqueReceiver)]],
    texture2d<float, access::read> opaque_geometry [[texture(9), function_constant(EnableOpaqueReceiver)]],
    texture2d<float, access::read> opaque_normal [[texture(10), function_constant(EnableOpaqueReceiver)]],
    texture2d<uint, access::read> opaque_identity [[texture(11), function_constant(EnableOpaqueReceiver)]],
    RAY_HISTORY_ARGS,
    RAY_KERNEL_ARGS,
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    device uint *pattern_hits [[buffer(BufferIndex_PatternHits), function_constant(EnablePatterns)]],
    device RayLightingSample *pattern_queue [[buffer(BufferIndex_PatternQueue), function_constant(EnablePatterns)]]
) {
    // A resumed sample is a listed sample whose first surface the hit driver evaluated.
    const uint listed = PatternResume ? thread_id.x : 0u;
    if (PatternResume && listed >= pattern_hits[PatternHitsListed]) return;
    const uint2 entry = PatternResume ? PatternHitEntry(pattern_hits, listed) : uint2(0u);
    const uint lobes = RayLightingLobeCount(EnableTransmission, TraceDiffuse);
    const uint index = PatternResume ? entry.x / lobes : thread_id.x, lobe = PatternResume ? entry.x % lobes : thread_id.y;
    if (index >= work[0]) return;
    const uint2 pixel = RayWorkPixel(work, index, visibility.get_width());
    if (!RayGuideDeferred(guide_image.read(pixel))) return;
    const uint queue_index = lobes * index + lobe;
    const RayLightingSample sample = queue[queue_index];
    const float4 ray = sample.DirectionPdf, origin = sample.OriginFootprint, position = sample.PositionFlags;
    const uint flags = as_type<uint>(position.w);
    const bool active = (flags & RaySampleActive) != 0u, valid = (flags & RaySampleValid) != 0u, glass = (flags & RaySampleGlass) != 0u;
    float3 incoming(0);
    float distance = active ? 1e10f : 0.0f;
    float4 hit_guide(0, 0, active ? -1.0f : -2.0f, 0);
    if (valid) {
        const Scene scene{bindless, view, theme, workspace};
        RayScene rays{acceleration, instances, triangles, environment, lut, feedback, history_geometry, history_normal, history_material,
            motion, pc.PreviousViewProj.Unpack(), pc.HistoryValid != 0u, pc.ApplyMotion != 0u, pc.PreviousClipToWorld.Unpack()};
        if (EnableOpaqueReceiver) {
            rays.OpaqueRadiance = opaque_radiance;
            rays.OpaqueGeometry = opaque_geometry;
            rays.OpaqueNormal = opaque_normal;
            rays.OpaqueIdentity = opaque_identity;
            rays.ClipToWorld = pc.ClipToWorld.Unpack();
            rays.OpaqueValid = EnableTransmission && pc.OpaqueLayerActive != 0u && glass && lobe != 0u;
        }
        const RayMediumState camera_media = RayCameraState(rays);
        rays.OpaqueValid = rays.OpaqueValid && camera_media.Count != 0u;
        bool cached = false;
        const RayHit traced = PatternResume ? RayHit{PatternHitSurface(scene, rays.Geometry(), sample, entry.y), uint4(0u)} : EnableOpaqueReceiver ?
            TraceRayHit(scene, rays, origin.xyz, ray.xyz, glass && lobe != 0u, &incoming, &cached) :
            TraceRayHit(scene, rays, origin.xyz, ray.xyz, glass && lobe != 0u);
        const DecodedSurface hit = traced.Surface;
        const uint4 hit_identity = traced.Identity;
        // A first surface whose material computes inputs waits for the hit driver, and the resuming kernel shades the sample.
        if (EnablePatterns && !PatternResume && !cached && hit.Valid && BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[hit.V.MaterialIndex].Program.Index != InvalidOffset) {
            return DeferPatternHit(pattern_hits, pattern_queue[queue_index], queue_index, hit_identity);
        }
        if (!cached) incoming = TraceRayRadiance(scene, rays, hit, ray.xyz, origin.w, glass ? nullptr : &camera_media, hit.Valid ? length(hit.V.WorldPosition - position.xyz) : INFINITY, nullptr,
            PatternResume ? PatternRingWord(view, listed) : 0u);
        incoming *= view.Exposure;
        if (!glass) incoming *= RayMediumAttenuation(scene, camera_media, length(float3(view.CameraPosition) - position.xyz));
        if (hit.Valid) {
            distance = length(hit.V.WorldPosition - position.xyz);
            hit_guide = glass ? float4(EncodeRayNormal(hit.GeometricNormal), distance, float(hit.V.MaterialIndex)) : float4(0, 0, distance, 0);
        }
    }
    const float count = pc.SingleSample ? 1.0f : float(max(view.RaySamples, 1u));
    const float3 previous = pc.StagedSample ? (lobe ? transmission.read(pixel).rgb : radiance_image.read(pixel).rgb) : float3(0);
    const float4 result(previous + incoming * float(as_type<half>(ushort(flags >> 16u))) / count, 1);
    if (lobe) transmission.write(result, pixel);
    else radiance_image.write(result, pixel);
    if (pc.StagedSample == 0u) {
        if (lobe) transmission_guide.write(hit_guide, pixel);
        else hit_image.write(hit_guide, pixel);
        const uint step = view.RayStep, frame = view.RayFrame + pc.SampleOffset;
        const uint2 center = RayGridCenter(pixel / step, frame, step, uint2(visibility.get_width(), visibility.get_height()));
        if (all(pixel == center)) {
            if (lobe) {
                transmission_samples.write(ray, pixel / step);
                transmission_incident.write(float4(incoming, distance), pixel / step);
            } else {
                ray_samples.write(ray, pixel / step);
                incident.write(float4(incoming, distance), pixel / step);
            }
        }
    }
}

#include "RayReconstruct.h"
