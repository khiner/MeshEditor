/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal port of Blender 222853b0d7e eevee_ray_denoise.bsl.hh temporal_main and
 * bilateral_main.
 */
#include "EeveeDenoise.h"
#include "EeveeFilm.h"
#include "RayGuide.h"
#include "RayWork.h"
#include "RayHistory.h"
#include "RayDeformationSurface.h"
constant bool FilmLayer [[function_constant(0)]];
#include "gpu/RayFilmPushConstants.h"
#include "gpu/RayReconstructPushConstants.h"
using namespace eevee_denoise;

inline float4 history_sample(texture2d<float, access::read> history, texture2d<uint, access::read> history_material, float2 previous, float4 current_guide, LocalStatistics local, texture2d<float, access::read> geometry, texture2d<float, access::read> normals, uint slot, float3 world, float3 normal, float footprint, bool virtual_hit, float4x4 previous_inverse, texture2d<float, access::read> moments, thread float2 &variance_history) {
    const int2 extent = int2(history.get_width(), history.get_height());
    const int2 base = int2(floor(previous));
    const float2 fraction = fract(previous);
    float4 sum(0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            const int2 p = base + int2(x, y);
            if (any(p < 0) || any(p >= extent)) continue;
            if (history_material.read(uint2(p)).r != uint(current_guide.w) + 1u || !RayHistorySurface(RayHistoryGeometry(geometry, uint2(p), previous_inverse), normals.read(uint2(p)).xy, slot, world, normal, footprint, virtual_hit)) continue;
            const float2 weights = select(1.0f - fraction, fraction, bool2(x, y));
            const float weight = weights.x * weights.y;
            sum += float4(history.read(uint2(p)).rgb, 1) * weight;
            if (virtual_hit) variance_history += float2(moments.read(uint2(p)).r, 1) * weight;
        }
    sum.rgb = to_ycocg(sum.rgb);
    return sum * history_weight(sum.rgb, local);
}

[[max_total_threads_per_threadgroup(64)]] kernel void RayTemporal(texture2d<float, access::read> spatial [[texture(0)]], texture2d<float, access::read> guide [[texture(1)]], depth2d<float, access::read> depth [[texture(2)]], texture2d<float, access::read> info [[texture(3)]], texture2d<float, access::read> history [[texture(4)]], texture2d<uint, access::read> history_material [[texture(5)]], texture2d<float, access::read> moments [[texture(6)]], texture2d<float, access::write> output [[texture(7)]], texture2d<float, access::write> output_moments [[texture(9)]], texture2d<float, access::read> closure_weight [[texture(10)]], texture2d<float, access::read> current_geometry [[texture(11)]], texture2d<float, access::read> previous_geometry [[texture(12)]], texture2d<float, access::read> previous_normal [[texture(13)]], device const RayMotionTransform *motion [[buffer(10)]], device const RayDeformedSurface *deformation [[buffer(11)]], constant RayReconstructPushConstants &pc [[buffer(1)]], device const uint *work [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], uint2 local_id [[thread_position_in_threadgroup]]) {
    const uint2 size = uint2(guide.get_width(), guide.get_height());
    const uint2 pixel = RayTilePixel(work[group.x], local_id, size.x);
    threadgroup float4 neighborhood[100];
    threadgroup uint2 identities[100];
    if (pc.Denoise && pc.HistoryValid) {
        const int2 origin = int2(pixel) - int2(local_id) - 1;
        for (uint i = local_id.y * 8u + local_id.x; i < 100u; i += 64u) {
            const int2 p = origin + int2(i % 10u, i / 10u);
            float4 value(0);
            uint2 identity(0);
            if (all(p >= 0) && all(p < int2(size))) {
                const float4 ng = guide.read(uint2(p));
                if (RayLobeActive(ng, pc)) {
                    // Spatial is shared by the lobes. An unscheduled neighbor has zero radiance.
                    const bool empty = pc.SplitTiles && RayLobeEmpty(ng, pc.Lobe, closure_weight, uint2(p));
                    value = float4(empty ? float3(0) : to_ycocg(spatial.read(uint2(p)).rgb), 1);
                    identity = uint2(as_type<uint>(ng.w), as_type<uint>(current_geometry.read(uint2(p)).y));
                }
            }
            neighborhood[i] = value;
            identities[i] = identity;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (any(pixel >= size)) return;
    const float4 g = guide.read(pixel);
    const bool active = RayLobeActive(g, pc) && !RayLobeEmpty(g, pc.Lobe, closure_weight, pixel);
    float4 current = spatial.read(pixel);
    current.rgb = safe_color(current.rgb);
    if (!active) {
        output.write(float4(0), pixel);
        output_moments.write(float4(0), pixel);
        return;
    }
    float variance = info.read(pixel).x;
    // Coverage is forward-composited transport, not a deferred EEVEE closure.
    if (!pc.Denoise || !pc.HistoryValid || g.w < 0.0f) {
        output.write(current, pixel);
        output_moments.write(float4(variance, 0, 1, 0), pixel);
        return;
    }
    const float4 geometry = RayHistoryGeometry(current_geometry, pixel, pc.ClipToWorld.Unpack());
    const uint slot = (as_type<uint>(geometry.w) & 0x7fffffffu) - 1u;
    float3 previous_world = geometry.xyz, previous_n = DecodeRayNormal(g.xy);
    if (!RayPreviousPixelSurface(motion, slot, pc.ApplyMotion != 0u, deformation, pixel.y * size.x + pixel.x, previous_world, previous_n)) {
        output.write(current, pixel);
        output_moments.write(float4(variance, 0, 0, 0), pixel);
        return;
    }
    const float footprint = RayHistoryFootprint(pc.PreviousViewProj.Unpack(), previous_world, float(size.y));
    const int2 extent = int2(spatial.get_width(), spatial.get_height());
    float3 mean = to_ycocg(current.rgb), moment = mean * mean;
    float count = 1;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            const int2 p = int2(pixel) + int2(x, y);
            if (any(p < 0) || any(p >= extent)) continue;
            const uint index = uint(int(local_id.y) + y + 1) * 10u + uint(int(local_id.x) + x + 1);
            const float4 neighbor = neighborhood[index];
            if (neighbor.w == 0 || any(identities[index] != uint2(as_type<uint>(g.w), as_type<uint>(geometry.w)))) continue;
            const float3 c = neighbor.rgb;
            mean += c;
            moment += c * c;
            count += 1;
        }
    const LocalStatistics local = finish_statistics(mean, moment, count);
    float4 old(0);
    float2 variance_history(0);
    for (uint candidate = 0; candidate < 2; ++candidate) {
        const float z = candidate == 0 ? depth.read(pixel) : info.read(pixel).y;
        const float3 world = RayWorldPoint(float2(pixel), z, float2(extent), pc.ClipToWorld.Unpack());
        const float4 clip = pc.PreviousViewProj.Unpack() * float4(world + previous_world - geometry.xyz, 1);
        const float2 previous = pc.Moving ? ndc_to_uv(clip.xy / clip.w) * float2(extent) - 0.5f : float2(pixel);
        if (clip.w <= 0.0f || !all(isfinite(previous))) continue;
        old += history_sample(history, history_material, previous, g, local, previous_geometry, previous_normal, slot, previous_world, previous_n, footprint, candidate != 0u, pc.PreviousClipToWorld.Unpack(), moments, variance_history);
    }
    variance_history.x /= max(variance_history.y, 1e-20f);
    old.rgb /= max(old.w, 1e-20f);
    old.rgb = from_ycocg(clamp(old.rgb, local.clamp_min, local.clamp_max));
    const float mix_factor = temporal_mix(variance, old.w);
    current.rgb = mix(current.rgb, safe_color(old.rgb), mix_factor);
    variance = temporal_variance(variance, variance_history.x, variance_history.y > 0.0f);
    output.write(current, pixel);
    output_moments.write(float4(variance, 0, 1, 0), pixel);
}

kernel void RayBilateral(texture2d<float, access::read> radiance [[texture(0)]], texture2d<float, access::read> guide [[texture(1)]], depth2d<float, access::read> depth [[texture(2)]], texture2d<float, access::read_write> color [[texture(3)]], texture2d<float, access::read> moments [[texture(6)]], texture2d<float, access::read> closure_weight [[texture(10)]], constant RayReconstructPushConstants &pc [[buffer(1)]], device const uint *work [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], uint2 local_id [[thread_position_in_threadgroup]]) {
    const uint2 size = uint2(guide.get_width(), guide.get_height());
    const uint2 pixel = RayTilePixel(work[group.x], local_id, size.x);
    if (any(pixel >= size)) return;
    const float4 g = guide.read(pixel);
    if (!RayLobeActive(g, pc) || RayLobeEmpty(g, pc.Lobe, closure_weight, pixel)) return;
    float4 value = radiance.read(pixel);
    const float roughness = RayGuideDeferred(g) ? closure_weight.read(pixel).a : RayGuideRoughness(g), variance = moments.read(pixel).r;
    if (pc.Denoise && g.w >= 0.0f && roughness >= 0.05f && variance >= 0.1f) {
        const int2 extent = int2(radiance.get_width(), radiance.get_height());
        const float3 normal = DecodeRayNormal(g.xy);
        const float3 world = RayWorldPoint(float2(pixel), depth.read(pixel), float2(extent), pc.ClipToWorld.Unpack());
        const float size = mix(0.0f, 9.0f, saturate(roughness * 8.0f) * saturate(variance * 2.0f));
        const uint count = uint(mix(1.0f, 15.0f, saturate(variance * 2.0f)));
        const float2 random = fract(float2(noise(float2(pixel) + 0.5f)) + RaySampleOffset(pc.Frame));
        float3 sum = to_log(value.rgb);
        float weight_sum = 1;
        for (uint i = 0; i < count; ++i) {
            const int2 offset = int2(floor((fract(hammersley(i, count) + random) - 0.5f) * size + 0.5f));
            const int2 p = int2(pixel) + offset;
            if (any(p < 0) || any(p >= extent)) continue;
            const float4 ng = guide.read(uint2(p));
            if (!RayLobeActive(ng, pc) || ng.w != g.w) continue;
            const float3 neighbor = RayWorldPoint(float2(p), depth.read(uint2(p)), float2(extent), pc.ClipToWorld.Unpack());
            const float weight = planar_weight(normal, world, neighbor) * gaussian_weight(gaussian_factor(size, 1.5f), dot(float2(offset), float2(offset))) * angle_weight(normal, DecodeRayNormal(ng.xy));
            sum += to_log(radiance.read(uint2(p)).rgb) * weight;
            weight_sum += weight;
        }
        value.rgb = from_log(sum / weight_sum);
    }
    const float4 base = color.read(pixel);
    if (pc.FastGI && g.z >= 4.f && pc.Lobe == 0u) {
        const float maximum = pc.MaxRoughness, rough = g.z - 4.f;
        value.rgb *= pc.Reflections ? RayReflectionWeight(rough, maximum) : 0.f;
    }
    if (g.w < 0.0f) {
        const float hole = clamp(value.a, 0.0f, 1.0f);
        color.write(float4(value.rgb + hole * base.rgb, 1.0f - hole + hole * base.a), pixel);
    } else color.write(float4(base.rgb + value.rgb * (RayGuideDeferred(g) && pc.Lobe < 2u ? closure_weight.read(pixel).rgb : float3(1)), base.a), pixel);
}

inline void RayFilmGather(texture2d<float, access::read> color, uint2 pixel, constant RayFilmPushConstants &pc,
    bool skip_empty, thread float4 &sum, thread float &weight_sum) {
    using namespace eevee_film;
    const int2 extent(color.get_width(), color.get_height());
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            const float spatial_weight = pc.Weights[(y + 1) * 3 + x + 1];
            if (skip_empty && spatial_weight <= 0.0f) continue;
            const float4 c = color_space(color.read(uint2(clamp(int2(pixel) + int2(x, y), int2(0), extent - 1))));
            const float weight = spatial_weight * luma_weight(c.x);
            sum += c * weight;
            weight_sum += weight;
        }
}

// EEVEE sample_accum_combined and stationary store_combined; input is already exposed.
// PreviousWeight tracks spatial sample weights, not the luminance-weighted color sum.
kernel void RayAccumulate(texture2d<float, access::read> color [[texture(0)]], texture2d<float, access::read> accumulation [[texture(1)]], texture2d<float, access::write> output [[texture(2)]], texture2d<float, access::write> next_history [[texture(3)]], constant RayFilmPushConstants &pc [[buffer(1)]], uint2 pixel [[thread_position_in_grid]]) {
    using namespace eevee_film;
    float4 sum(0);
    float weight_sum = 0;
    RayFilmGather(color, pixel, pc, true, sum, weight_sum);
    if (pc.PreviousWeight > 0.0f) {
        const float4 history = color_space(accumulation.read(pixel));
        const float weight = pc.PreviousWeight * luma_weight(history.x);
        sum += history * weight;
        weight_sum += weight;
    }
    float4 value = sum / weight_sum;
    value.rgb = from_ycocg(value.rgb);
    if (value.a > .995f) value.a = 1.0f;
    if (any(isnan(value))) value = float4(0, 0, 0, 1);
    value = max(value, float4(0));
    next_history.write(value, pixel);
    output.write(value, pixel);
}

// EEVEE prepass velocity takes precedence over depth-derived camera motion.
inline bool RayFilmObjectMoving(RayMotionTransform motion) {
    if (motion.Valid == RayMotionDeformed) return true;
    if (motion.Valid != RayMotionRigid) return false;
    const float4x4 m = motion.CurrentToPrevious.Unpack();
    return any(m[0] != float4(1, 0, 0, 0)) || any(m[1] != float4(0, 1, 0, 0)) ||
        any(m[2] != float4(0, 0, 1, 0)) || any(m[3] != float4(0, 0, 0, 1));
}
inline float RayFilmDepth(uint2 pixel, depth2d<float, access::read> depth,
    texture2d<uint, access::read> materials, depth2d<float, access::read> layer_depth) {
    return FilmLayer && materials.read(pixel).r == 0u ? layer_depth.read(pixel) : depth.read(pixel);
}

// Film reads the previous image and writes a separate history image.
kernel void RayFilmReproject(
    texture2d<float, access::read> color [[texture(0)]], texture2d<float> history [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]], texture2d<float, access::write> next_history [[texture(3)]],
    texture2d<float, access::read> geometry [[texture(4)]], texture2d<float, access::read> previous_geometry [[texture(5)]],
    texture2d<float, access::read> normals [[texture(6)]], texture2d<float, access::read> previous_normals [[texture(7)]],
    depth2d<float, access::read> depth [[texture(8)]], texture2d<uint, access::read> materials [[texture(9)]], texture2d<uint, access::read> previous_materials [[texture(10)]],
    texture2d<float, access::read> surface_guide [[texture(13)]],
    texture2d<uint, access::read> layer_visibility [[texture(11)]], depth2d<float, access::read> layer_depth [[texture(12)]],
    device const BindlessSet &bindless [[buffer(0)]], constant SceneViewUBO &view [[buffer(2)]],
    constant ViewportThemeUbo &theme [[buffer(3)]], constant WorkspaceLights &workspace [[buffer(4)]],
    device const packed_float3 *previous_positions [[buffer(12)]],
    device const RayMotionTransform *motion [[buffer(10)]], device const RayDeformedSurface *deformation [[buffer(11)]], constant RayFilmReprojectPushConstants &pc [[buffer(1)]], uint2 pixel [[thread_position_in_grid]]
) {
    using namespace eevee_film;
    const int2 extent(color.get_width(), color.get_height());
    float4 source(0);
    float weight_sum = 0;
    RayFilmGather(color, pixel, pc.Filter, false, source, weight_sum);
    source /= max(weight_sum, 1e-8f);
    float4 value = float4(from_ycocg(source.rgb), source.a);
    uint2 nearest = pixel;
    float min_depth = RayFilmDepth(pixel, depth, materials, layer_depth);
    const int2 corners[4] = {int2(-2, -2), int2(2, -2), int2(-2, 2), int2(2, 2)};
    for (uint i = 0; i < 4; ++i) {
        const uint2 p = uint2(clamp(int2(pixel) + corners[i], int2(0), extent - 1));
        const float z = RayFilmDepth(p, depth, materials, layer_depth);
        if (z < min_depth) {
            min_depth = z;
            nearest = p;
        }
    }
    const float4 g = RayHistoryGeometry(geometry, nearest, pc.CurrentGeometryClipToWorld.Unpack());
    const uint tag = as_type<uint>(g.w), slot = (tag & 0x7fffffffu) - 1u;
    float3 world = g.xyz, normal = DecodeRayNormal(normals.read(nearest).xy);
    bool valid = (tag & 0x7fffffffu) != 0u && materials.read(nearest).r != 0u;
    if (valid) valid = RayPreviousPixelSurface(motion, slot, pc.ApplyMotion != 0u, deformation, nearest.y * uint(extent.x) + nearest.x, world, normal);
    float2 history_pixel(0);
    bool layer_motion = false;
    if (FilmLayer && materials.read(nearest).r == 0u) {
        layer_motion = true;
        bool object_velocity = false;
        bool layer_valid = true;
        float3 current = g.xyz, previous = current;
        if (pc.ApplyMotion) {
            const auto resolved = ResolveVisibilityId(layer_visibility.read(nearest).r, bindless, view, theme, workspace, pc.Visibility);
            if (resolved.Valid && RayFilmObjectMoving(motion[resolved.Draw.FirstInstance])) {
                current = RayWorldPoint(float2(nearest), layer_depth.read(nearest), float2(extent), pc.CurrentGeometryClipToWorld.Unpack());
                const auto transform = motion[resolved.Draw.FirstInstance];
                if (transform.Valid == RayMotionDeformed) {
                    if (pc.Deformation && MeshletPrimitiveTopology(resolved.Meshlet) == uint(MeshPrimitiveTopology::Triangle)) {
                        const auto surface = RayPreviousVisibleSurface(nearest, resolved, bindless, view, theme, workspace, transform, previous_positions);
                        previous = surface.PositionValid.xyz;
                        layer_valid = surface.PositionValid.w > 0.0f;
                    } else layer_valid = false;
                } else previous = (transform.CurrentToPrevious.Unpack() * float4(current, 1)).xyz;
                object_velocity = true;
            }
        }
        const bool background = !object_velocity && min_depth == 1.0f;
        if (!object_velocity) {
            current = RayWorldPoint(float2(nearest), min_depth, float2(extent), pc.CurrentGeometryClipToWorld.Unpack());
            if (background) current -= RayWorldPoint(float2(nearest), 0.0f, float2(extent), pc.CurrentGeometryClipToWorld.Unpack());
            previous = current;
        }
        const float4 current_clip = pc.CurrentViewProj.Unpack() * float4(current, background ? 0.0f : 1.0f);
        const float4 old_clip = pc.PreviousViewProj.Unpack() * float4(previous, background ? 0.0f : 1.0f);
        history_pixel = float2(pixel);
        if (current_clip.w != 0.0f && old_clip.w > 0.0f)
            history_pixel += (old_clip.xy / old_clip.w - current_clip.xy / current_clip.w) * float2(.5f, -.5f) * float2(extent);
        else layer_valid = layer_valid && background && current_clip.w == 0.0f && old_clip.w == 0.0f;
        valid = layer_valid && all(isfinite(history_pixel)) && all(history_pixel >= 0) && all(history_pixel < float2(extent));
        if (valid) {
            const float4 old_geometry_clip = pc.PreviousGeometryViewProj.Unpack() * float4(previous, background ? 0.0f : 1.0f);
            const float2 old_pixel = old_geometry_clip.w != 0.0f ?
                ndc_to_uv(old_geometry_clip.xy / old_geometry_clip.w) * float2(extent) : history_pixel;
            valid = all(isfinite(old_pixel)) && all(old_pixel >= 0) && all(old_pixel < float2(extent));
            // The sampled film must contain the same front layer as the current composite.
            if (valid) valid = as_type<uint>(previous_geometry.read(uint2(old_pixel)).y) == tag;
        }
    }
    if (!layer_motion && valid) {
        const float4 current_clip = pc.CurrentViewProj.Unpack() * float4(g.xyz, 1);
        const float4 old_clip = pc.PreviousViewProj.Unpack() * float4(world, 1);
        const float4 geometry_clip = pc.PreviousGeometryViewProj.Unpack() * float4(world, 1);
        history_pixel = float2(pixel) + (old_clip.xy / old_clip.w - current_clip.xy / current_clip.w) * float2(.5f, -.5f) * float2(extent);
        const float2 previous_pixel = ndc_to_uv(geometry_clip.xy / geometry_clip.w) * float2(extent);
        valid = current_clip.w > 0 && old_clip.w > 0 && geometry_clip.w > 0 && geometry_clip.z >= 0 && geometry_clip.z <= geometry_clip.w && all(isfinite(history_pixel)) && all(history_pixel >= 0) && all(history_pixel < float2(extent)) && all(previous_pixel >= 0) && all(previous_pixel < float2(extent));
        if (valid) {
            const uint2 p = uint2(previous_pixel);
            const float4 old_geometry = RayHistoryGeometry(previous_geometry, p, pc.PreviousGeometryClipToWorld.Unpack());
            const float footprint = RayHistoryFootprint(pc.PreviousGeometryViewProj.Unpack(), world, float(extent.y));
            valid = previous_materials.read(p).r == materials.read(nearest).r && ((as_type<uint>(old_geometry.w) ^ tag) & 0x80000000u) == 0u && RayHistorySurface(old_geometry, previous_normals.read(p).xy, slot, world, normal, footprint);
        }
    }
    if (valid) {
        const float4 old = color_space(sample_catmull_rom(history, history_pixel));
        float4 low, high;
        const float4 guide = pc.TransparentSurfaces ? surface_guide.read(pixel) : float4(0);
        if (RayGuideSmoothGlass(guide) || (FilmLayer && guide.w < 0.0f))
            variance_bounds(color, int2(pixel), low, high);
        else
            neighbor_bounds(color, int2(pixel), low, high);
        value = resolve(source, old, low, high, length(history_pixel - float2(pixel)));
    }
    value.rgb = eevee_denoise::safe_color(value.rgb);
    value.a = saturate(value.a);
    next_history.write(value, pixel);
    output.write(value, pixel);
}
