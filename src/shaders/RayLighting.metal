#pragma once
#include "pbr.metal"
#include "gpu/RayLightingPushConstants.h"

inline bool DeferredOpaque(device const PBRMaterial &m) {
    return !m.Unlit && m.AlphaMode != MaterialAlphaMode::Blend && m.Transmission.Factor == 0.0f &&
        m.Clearcoat.Factor == 0.0f && m.Anisotropy.Strength == 0.0f && all(float3(m.Sheen.ColorFactor) == float3(0)) &&
        m.Iridescence.Factor == 0.0f && m.DiffuseTransmission.Factor == 0.0f;
}
inline eevee_opaque::Closure PrepareOpaque(const thread PbrContext &ctx, const thread RayScene &rays) {
    device const PBRMaterial &m = ctx.S.Materials(ctx.S.View.MaterialSlot)[ctx.In.MaterialIndex];
    float3 base = float3(m.BaseColorFactor.rgb) * ctx.In.VertexColor.rgb;
    if (m.BaseColorTexture.Slot != InvalidSlot) base *= ctx.SampleTexture(m.BaseColorTexture).rgb;
    const float2 mr = saturate(ctx.MetallicRoughness(m));
    const float4 specular = ctx.SpecularColorStrength(m);
    const float ior_f0 = (m.Ior - 1.0f) / (m.Ior + 1.0f);
    float ao = 1.0f;
    if (m.OcclusionTexture.Slot != InvalidSlot) ao += m.OcclusionStrength * (ctx.SampleTexture(m.OcclusionTexture).r - 1.0f);
    return eevee_opaque::prepare(rays.Lut, ctx.GetNormalInfo<false>(m).n, ctx.ViewDirection(), mr.y, base, mr.x, min(specular.rgb * ior_f0 * ior_f0, float3(1)), specular.a, ao);
}
template<PbrMode Mode, bool Prepare = false>
inline float4 ShadeRayPixel(const thread Scene &scene, const thread RayScene &rays, const thread DecodedSurface &decoded,
    constant RayLightingPushConstants &pc, thread float3 &radiance, thread float4 &guide, thread RayClosureWeights *weights,
    thread float4 *hit_guide = nullptr, thread float *camera_hole = nullptr, thread RayReuseSample *sample = nullptr
#ifdef FILTER_AREA_SHADOWS
    , uint lane = 0u, uint visibility_id = 0u
#endif
) {
    PbrContext ctx{scene, decoded.V};
    ctx.Sample = sample;
    ctx.SampleCount = Mode == PbrMode::Indirect && pc.SingleSample ? 1u : scene.View.RaySamples;
    ctx.SampleOffset = pc.SampleOffset;
    ctx.SetSurface(decoded);
    ctx.RayTracing = true;
    const RayMediumState camera_media = RayCameraState(rays);
    ctx.Media = &camera_media;
    if (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle)) {
        guide = float4(0, 0, -1, 0);
        ctx.GeometricNormal = ShadingWorldNormal(decoded.V);
        const float4 color = ShadePbr<PbrMode::Secondary>(decoded.V, decoded.Topology, decoded.PointCoord, scene, scene.View, ctx, &rays);
        const float3 attenuation = RayMediumAttenuation(scene, camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition));
        return float4(color.rgb * attenuation, color.a);
    }
    device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[decoded.V.MaterialIndex];
    const bool blended = EnableTransmission && EnableAlphaBlend && material.AlphaMode == MaterialAlphaMode::Blend;
    if (blended || (EnableTransmission && !material.Unlit && material.Transmission.Factor > 0.0f)) {
        // Glass transport is independent of the opaque reflection/GI controls.
        const float roughness = ctx.MetallicRoughness(material).y;
        guide = float4(EncodeRayNormal(decoded.GeometricNormal), blended ? 0.0f : 2.0f + clamp(roughness, 0.0f, 1.0f), blended ? -1.0f - float(decoded.V.MaterialIndex) : float(decoded.V.MaterialIndex));
        if (!blended) {
            if constexpr (Prepare) *weights = GlassWeights(PrepareGlass(ctx, rays));
            return float4(0, 0, 0, 1);
        }
        if constexpr (Mode == PbrMode::Indirect) {
            float hole = 0.0f;
            radiance = scene.View.Exposure * TraceRayRadiance(scene, rays, decoded, -ctx.ViewDirection(), max(length(decoded.PositionDx), length(decoded.PositionDy)), &camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition), &hole);
            if (camera_hole) *camera_hole = hole;
        }
        return float4(0, 0, 0, 1);
    }
    if constexpr (Prepare) return float4(0);
#ifdef FILTER_AREA_SHADOWS
    if (Mode == PbrMode::Direct) ctx.ShadowNeighbors = ShadowNeighborhood::make(visibility_id, uint2(decoded.V.Position.xy), lane);
#endif
    const bool deferred = DeferredOpaque(material);
    eevee_opaque::Closure opaque{};
    if constexpr (Mode == PbrMode::Direct)
        if (deferred) ctx.OpaqueClosure = &opaque;
    float roughness = -1.0f;
    const float4 base = ShadePbr<Mode>(decoded.V, uint(MeshPrimitiveTopology::Triangle), decoded.PointCoord, scene, scene.View, ctx, &rays, &radiance, &roughness, hit_guide);
    if (!TraceDiffuse && !(scene.View.RayFastGI && deferred) && (!TraceReflections || RayReflectionWeight(roughness, scene.View.RayMaxRoughness) == 0.0f)) roughness = -1.0f;
    guide = float4(EncodeRayNormal(decoded.GeometricNormal), roughness, float(decoded.V.MaterialIndex));
    if constexpr (Mode == PbrMode::Direct) {
        if (deferred && roughness >= 0.0f) {
            guide = float4(EncodeRayNormal(opaque.N), 4.0f + roughness, float(decoded.V.MaterialIndex));
            weights->Reflection = float4(opaque.Reflection * (scene.View.RayFastGI ? 1.0f : TraceReflections ? RayReflectionWeight(roughness, scene.View.RayMaxRoughness) : 0.0f), roughness);
            weights->Transmission = float4((TraceDiffuse || scene.View.RayFastGI) ? opaque.Diffuse : float3(0), 1.0f);
        }
    }
    const float3 attenuation = RayMediumAttenuation(scene, camera_media, distance(float3(scene.View.CameraPosition), decoded.V.WorldPosition));
    radiance *= attenuation;
    if (sample) sample->IncidentDistance.rgb *= attenuation;
    return float4(base.rgb * attenuation, base.a);
}

kernel void PbrOpaqueLayer(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<uint, access::read> primary_visibility [[texture(1)]],
    depth2d<float, access::read> depth [[texture(20)]],
    texture2d<float, access::write> radiance [[texture(3)]],
    texture2d<float, access::write> geometry [[texture(9)]],
    texture2d<float, access::write> normal [[texture(10)]],
    texture2d<uint, access::write> identity [[texture(11)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pixel.x >= visibility.get_width() || pixel.y >= visibility.get_height()) return;
    identity.write(uint4(0), pixel);
    geometry.write(float4(0), pixel);
    const uint2 primary_pixel = min(uint2((float2(pixel) + 0.5f) * float2(primary_visibility.get_width(), primary_visibility.get_height()) / float2(visibility.get_width(), visibility.get_height())), uint2(primary_visibility.get_width(), primary_visibility.get_height()) - 1u);
    if (visibility.read(pixel).r == primary_visibility.read(primary_pixel).r) return;
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    if (!decoded.Valid || (NonTriangleTopology && decoded.Topology != uint(MeshPrimitiveTopology::Triangle))) return;
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    PbrContext ctx{scene, decoded.V};
    ctx.SetSurface(decoded);
    ctx.Secondary = ctx.Opaque = true;
    ctx.Outgoing = view.ScreenPixelScale < 0.0f ? transpose(view.ViewRotation.Unpack()) * float3(0, 0, 1) : normalize(float3(view.CameraPosition) - decoded.V.WorldPosition);
    ctx.Footprint = float(visibility.get_width()) / float(primary_visibility.get_width());
    radiance.write(ShadePbr<PbrMode::Secondary>(decoded.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, view, ctx, &rays), pixel);
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
    texture2d<float, access::write> history_geometry [[texture(17)]],
    texture2d<float, access::write> history_normal [[texture(18)]],
    depth2d<float, access::read> history_depth [[texture(20)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
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

    float3 radiance{};
    float4 guide{};
    RayClosureWeights weights{};
    const float4 base = PreparePrimary ?
        ShadeRayPixel<PbrMode::Direct, true>(scene, rays, decoded, pc, radiance, guide, &weights) :
        ShadeRayPixel<PbrMode::Direct>(scene, rays, decoded, pc, radiance, guide, &weights
#ifdef FILTER_AREA_SHADOWS
            , nullptr, nullptr, nullptr, lane, visibility_id
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
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
    texture2d<float, access::write> glass_shape [[texture(21)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    if (RayCameraState(rays).Count != 0u || !RayGuideGlass(guide_image.read(pixel))) return;
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    const Scene scene{bindless, view, theme, workspace};

    PbrContext ctx{scene, decoded.V};
    ctx.SetSurface(decoded);
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
    texture2d<float, access::read> feedback [[texture(16)]],
    texture2d<float, access::read> history_geometry [[texture(17)]],
    texture2d<float, access::read> history_normal [[texture(18)]],
    texture2d<uint, access::read> history_material [[texture(19)]],
    device const RayMotionTransform *motion [[buffer(10)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (index >= work[0]) return;
    const uint2 pixel = RayWorkPixel(work, index, visibility.get_width());
    if (RayGuideDeferred(guide_image.read(pixel))) return;
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut, feedback, history_geometry, history_normal, history_material,
        motion, pc.PreviousViewProj.Unpack(), pc.HistoryValid != 0u, pc.ApplyMotion != 0u, pc.PreviousClipToWorld.Unpack()};
    const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
    float3 radiance{};
    float4 guide{};
    float4 hit_guide{0, 0, -2, 0};
    float hole = 0.0f;
    RayReuseSample sample{};
    ShadeRayPixel<PbrMode::Indirect>(scene, rays, decoded, pc, radiance, guide, nullptr, &hit_guide, &hole, &sample);
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

kernel void PbrRayGenerate(
    uint index [[thread_position_in_grid]],
    device RayLightingSample *queue [[buffer(11)]],
    device const uint *work [[buffer(9)]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    texture2d<float, access::read> guide_image [[texture(1)]],
    texture2d<float, access::write> direct_lighting [[texture(14)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
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
    ctx.SetSurface(decoded);
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
            ctx.GlassFraction = cl.Fraction;
            if (cl.Fraction < 1.0f) direct = ShadePbr<PbrMode::GlassSurface>(decoded.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, view, ctx, &rays).rgb;
            else if ((EnableSceneLights && view.UseSceneLightsRender && view.LightCount) || any(float3(material.EmissiveFactor) > 0.0f)) direct = ShadePbr<PbrMode::GlassDirect>(decoded.V, uint(MeshPrimitiveTopology::Triangle), float2(0), scene, view, ctx, &rays).rgb;
            direct_lighting.write(float4(direct, 1), pixel);
        }
        for (uint lobe = 0u; lobe < 2u; ++lobe) {
            const float2 noise = fract((lobe ? blue.zw : blue.xy) + RaySampleOffset(frame));
            float3 origin;
            const float4 ray = eevee_glass::generate(cl, lobe, noise, origin);
            const bool valid = ray.w > 0.0f && dot(ray.xyz, ray.xyz) > 0.5f;
            samples[lobe] = {ray, float4(RayOrigin(origin, cl.N, ray.xyz), footprint), float4(origin, as_type<float>(5u | (valid ? 2u : 0u)))};
        }
    } else {
        const auto cl = PrepareOpaque(ctx, rays);
        const float reflection = TraceReflections ? RayReflectionWeight(cl.Roughness, view.RayMaxRoughness) : 0.0f;
        if (EnableTransmission && pc.StagedSample == 0u) direct_lighting.write(float4(0, 0, 0, 1), pixel);
        for (uint lobe = 0u; lobe < (TraceDiffuse ? 2u : 1u); ++lobe) {
            if (all((lobe ? cl.Diffuse : cl.Reflection * reflection) == float3(0))) continue;
            const float2 noise = fract((lobe ? blue.zw : blue.xy) + RaySampleOffset(frame));
            const float4 ray = eevee_opaque::generate(cl.N, ctx.ViewDirection(), cl.Roughness, lobe, noise);
            const bool valid = ray.w > 0.0f && dot(decoded.GeometricNormal, ray.xyz) * dot(decoded.GeometricNormal, ctx.ViewDirection()) > 0.0f;
            samples[lobe] = {ray, float4(RayOrigin(decoded.V.WorldPosition, decoded.GeometricNormal, ray.xyz), footprint), float4(decoded.V.WorldPosition, as_type<float>(1u | (valid ? 2u : 0u)))};
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
    texture2d<float, access::read> feedback [[texture(16)]],
    texture2d<float, access::read> history_geometry [[texture(17)]],
    texture2d<float, access::read> history_normal [[texture(18)]],
    texture2d<uint, access::read> history_material [[texture(19)]],
    texture2d<float, access::read> opaque_radiance [[texture(3), function_constant(EnableOpaqueReceiver)]],
    texture2d<float, access::read> opaque_geometry [[texture(9), function_constant(EnableOpaqueReceiver)]],
    texture2d<float, access::read> opaque_normal [[texture(10), function_constant(EnableOpaqueReceiver)]],
    texture2d<uint, access::read> opaque_identity [[texture(11), function_constant(EnableOpaqueReceiver)]],
    device const RayMotionTransform *motion [[buffer(10)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const float *environment [[buffer(8)]],
    texture2d_array<float> lut [[texture(15)]],
    constant RayLightingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint index = thread_id.x, lobe = thread_id.y;
    if (index >= work[0]) return;
    const uint2 pixel = RayWorkPixel(work, index, visibility.get_width());
    if (!RayGuideDeferred(guide_image.read(pixel))) return;
    const RayLightingSample sample = queue[RayLightingLobeCount(EnableTransmission, TraceDiffuse) * index + lobe];
    const float4 ray = sample.DirectionPdf, origin = sample.OriginFootprint, position = sample.PositionFlags;
    const uint flags = as_type<uint>(position.w);
    const bool active = (flags & 1u) != 0u, valid = (flags & 2u) != 0u, glass = (flags & 4u) != 0u;
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
        const DecodedSurface hit = EnableOpaqueReceiver ?
            TraceSurface(scene, rays, origin.xyz, ray.xyz, glass && lobe != 0u, &incoming, &cached) :
            TraceSurface(scene, rays, origin.xyz, ray.xyz, glass && lobe != 0u);
        if (!cached) incoming = TraceRayRadiance(scene, rays, hit, ray.xyz, origin.w, glass ? nullptr : &camera_media, hit.Valid ? length(hit.V.WorldPosition - position.xyz) : INFINITY);
        incoming *= view.Exposure;
        if (!glass) incoming *= RayMediumAttenuation(scene, camera_media, length(float3(view.CameraPosition) - position.xyz));
        if (hit.Valid) {
            distance = length(hit.V.WorldPosition - position.xyz);
            hit_guide = glass ? float4(EncodeRayNormal(hit.GeometricNormal), distance, float(hit.V.MaterialIndex)) : float4(0, 0, distance, 0);
        }
    }
    const float count = pc.SingleSample ? 1.0f : float(max(view.RaySamples, 1u));
    const float3 previous = pc.StagedSample ? (lobe ? transmission.read(pixel).rgb : radiance_image.read(pixel).rgb) : float3(0);
    const float4 result(previous + incoming / count, 1);
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
