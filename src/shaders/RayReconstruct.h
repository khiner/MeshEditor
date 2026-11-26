/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal integration of Blender 222853b0d7e eevee_ray_denoise.bsl.hh spatial_main.
 */
#pragma once
kernel void RayReconstruct(
    texture2d<float, access::read> radiance [[texture(0)]],
    texture2d<float, access::read> guide [[texture(1)]],
    depth2d<float, access::read> depth [[texture(2)]],
    texture2d<float, access::write> output [[texture(3)]],
    texture2d<float, access::read> hit [[texture(4)]],
    texture2d<float, access::read> ray_samples [[texture(5)]],
    texture2d<float, access::read> incident [[texture(6)]],
    texture2d<float, access::read> validity [[texture(7)]],
    texture2d<float, access::write> info [[texture(8)]],
    texture2d<uint, access::read> visibility [[texture(9)]],
    texture2d<float, access::read> closure_weight [[texture(10)]],
    texture2d<float, access::read> closure_normal [[texture(11)]],
    texture2d<float, access::read> glass_shape [[texture(12)]],
    texture2d<float, access::read> coat_weight [[texture(13), function_constant(TraceCoat)]],
    texture2d<float, access::read> coat_normal [[texture(14), function_constant(TraceCoat)]],
    texture2d<float, access::read> diffuse_transmission_weight [[texture(16), function_constant(TraceDiffuseTransmission)]],
    texture2d<float, access::read> sheen_weight [[texture(17), function_constant(TraceSheen)]],
    RAY_KERNEL_ARGS,
    constant RayReconstructPushConstants &pc [[buffer(1)]],
    device const uint *work [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]
) {
    using namespace eevee_denoise;
    const uint2 size = uint2(guide.get_width(), guide.get_height());
    const uint2 pixel = RayTilePixel(work[group.x], local, size.x);
    if (any(pixel >= size)) return;
    const float4 g = guide.read(pixel);
    if (!RayLobeActive(g, pc)) {
        output.write(float4(0), pixel);
        info.write(float4(0), pixel);
        return;
    }
    if (RayLobeEmpty(g, pc.Lobe, closure_weight, pixel)) {
        output.write(float4(0), pixel);
        info.write(float4(0, depth.read(pixel), 0, 0), pixel);
        return;
    }
    const float4 own = validity.read(pixel).a > 0.0f ? radiance.read(pixel) : float4(0);
    if (g.w < 0.0f) {
        output.write(float4(own.rgb, clamp(own.a - 1.0f, 0.0f, 1.0f)), pixel);
        info.write(float4(0, depth.read(pixel), 0, 0), pixel);
        return;
    }
    const int2 extent = int2(radiance.get_width(), radiance.get_height());
    const float roughness = RayGuideDeferred(g) ? closure_weight.read(pixel).a : RayGuideRoughness(g);
    const int step = roughness <= 0.08f && !RayGuideDeferred(g) ? 1 : int(pc.Step);
    const int2 offset = int2(RayGridOffset(pc.Frame, uint(step)));
    const float2 grid = (float2(pixel) - float2(offset)) / float(step);
    const float2 random = fract(lut.read(pixel % 64u, 0u).zw + RaySampleOffset(pc.Frame));
    const int2 center = min(clamp(int2(floor(grid)) + int2(fract(grid) > random), int2(0), (extent - 1) / step) * step + offset, extent - 1);
    const float3 world = RayWorldPoint(float2(pixel), depth.read(pixel), float2(extent), pc.ClipToWorld.Unpack());
    const float3 normal = DecodeRayNormal(g.xy);
    const Scene scene{bindless, view, theme, workspace};
    const RayScene rays{acceleration, instances, triangles, environment, lut};
    // A reflection ray that carries several lobes reconstructs by their values at every roughness.
    const bool ratio = pc.Denoise != 0u && ((TraceExtraLobes && pc.Lobe == 0u && RayGuideSharedReflection(g)) || roughness > (RayGuideGlass(g) ? 0.02f : 0.08f));
    float3 sum(0), moment(0);
    float weight_sum = 0, closest = 1e10f;
    if (ratio) {
        const float factor = saturate(roughness * 8.0f);
        uint count = 1u + uint(floor(15.0f * factor + 0.5f));
        const float min_radius = step > 1 ? 1.5f : 0.0f;
        if (step > 1) count = max(count, 5u);
        const float radius = 8.0f * sqrt(factor);
        const float aspect = 1.0f - 0.5f * saturate(min(roughness * 12.0f, 2.0f - roughness * 5.0f));
        const float3 view_normal = view.ViewRotation.Unpack() * normal;
        const float2 up = dot(view_normal.xy, view_normal.xy) > 1e-12f ? normalize(view_normal.xy) : float2(1, 0);
        const float2x2 rotation = float2x2(up * clamp(radius, min_radius, 7.0f), float2(-up.y, up.x) * clamp(radius * aspect, min_radius, 7.0f));
        const float bias = max(0.2f, saturate((0.75f - roughness) / (0.75f - 0.3f)));
        RaySpatialReuse reuse{guide, ray_samples, incident, validity, depth};
        reuse.Extent = extent;
        reuse.Center = center;
        reuse.GridOffset = offset;
        reuse.Step = step;
        reuse.Count = count;
        reuse.Rotation = rotation;
        reuse.Bias = bias;
        reuse.Random = random.x;
        reuse.Material = g.w;
        reuse.World = world;
        reuse.ClipToWorld = pc.ClipToWorld.Unpack();
        reuse.Closest = 1e10f;
        if (g.z >= 4.0f) {
            const float3 v = view.ScreenPixelScale < 0.0f ? transpose(view.ViewRotation.Unpack()) * float3(0, 0, 1) : normalize(float3(view.CameraPosition) - world);
            if (TraceExtraLobes) {
                // Each sample weighs by the sum of the ray's lobe values, and carries their color relative to the ray's weight.
                const float3 total = closure_weight.read(pixel).rgb;
                const DeferredRay ray = DeferredLobes(view, pc.Lobe, pixel, g, v, closure_weight, closure_normal, coat_weight, coat_normal, diffuse_transmission_weight, sheen_weight);
                constexpr float3 Average(1.0f / 3.0f);
                float3 carried(0);
                for (uint k = 0u; k < ray.Count; ++k) carried += ray.Weights[k];
                const float3 scale = select(float3(0), dot(carried, Average) / total, total > 0.0f);
                for (uint i = 0; i < count; ++i) {
                    float3 direction;
                    if (!reuse.Sample(i, direction)) continue;
                    const float3 value = ray.Value(v, direction);
                    const float luminance = dot(value, Average);
                    if (luminance > 0.0f) reuse.Accumulate(reuse.Radiance * value / luminance * scale, luminance);
                }
            } else {
                for (uint i = 0; i < count; ++i) {
                    float3 direction;
                    if (reuse.Sample(i, direction)) reuse.Accumulate(reuse.Radiance, DeferredSharedPdf({DecodeRayNormal(closure_normal.read(pixel).xy), RayGuideRoughness(g), pc.Lobe ? DeferredKind::Diffuse : DeferredKind::Ggx, 0}, v, direction));
                }
            }
        } else if (RayGuideGlass(g)) {
            // EEVEE reconstruction reads prepared closure data instead of evaluating the material again.
            const float4 shape = glass_shape.read(pixel);
            eevee_glass::Closure cl{};
            cl.N = DecodeRayNormal(shape.xy);
            cl.V = view.ScreenPixelScale < 0.0f ? transpose(view.ViewRotation.Unpack()) * float3(0, 0, 1) : normalize(float3(view.CameraPosition) - world);
            cl.Position = world;
            cl.Roughness = shape.z;
            cl.Thickness = shape.w;
            cl.Ior = max(scene.Materials(view.MaterialSlot)[uint(g.w)].Ior, 1.0f);
            for (uint i = 0; i < count; ++i) {
                float3 direction;
                if (reuse.Sample(i, direction)) reuse.Accumulate(reuse.Radiance, eevee_glass::pdf(cl, pc.Lobe, direction));
            }
        } else {
            const auto decoded = DecodeVisibility(float2(pixel) + 0.5f, visibility, bindless, view, theme, workspace, pc.Visibility);
            PbrContext ctx{scene, decoded.V};
            ctx.SetSurface(decoded, PixelPatternWord(view, decoded.Topology, pixel));
            ctx.RayTracing = true;
            ShadeReuse(scene, ctx, rays, reuse);
        }
        sum = reuse.Sum;
        moment = reuse.Moment;
        weight_sum = reuse.Weight;
        closest = reuse.Closest;
    } else {
        // EEVEE skip-denoise bilateral footprint in logarithmic light.
        // The regular and cell-center samples form a square grid rotated by 45 degrees.
        const bool quincunx = pc.QuincunxGlass && RayGuideSmoothGlass(g);
        const float2 coordinates = quincunx ? float2(grid.x + grid.y, grid.x - grid.y) : grid;
        const int2 cell = int2(floor(coordinates));
        const float2 f = fract(coordinates);
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                const int2 index = cell + int2(x, y);
                const int2 point = quincunx ? int2(index.x + index.y, index.x - index.y) * (step / 2) : index * step;
                const int2 p = clamp(point + offset, int2(0), extent - 1);
                const float4 sample = radiance.read(uint2(p)), ng = guide.read(uint2(p));
                if (validity.read(uint2(p)).a == 0.0f || ng.w != g.w || ng.z < 0.0f) continue;
                const float3 neighbor = RayWorldPoint(float2(p), depth.read(uint2(p)), float2(extent), pc.ClipToWorld.Unpack());
                const float2 weights = select(1.0f - f, f, bool2(x, y));
                const float weight = weights.x * weights.y * max(1e-4f, planar_weight(normal, world, neighbor) * angle_weight(normal, DecodeRayNormal(ng.xy)));
                const float3 c = to_log(sample.rgb);
                sum += c * weight;
                moment += c * c * weight;
                weight_sum += weight;
                const float d = hit.read(uint2(p)).z;
                if (weight > 0.0f && d >= 0.0f) closest = min(closest, d);
            }
    }
    if (weight_sum <= 0.0f) {
        // Full-resolution boundary samples are valid even if their regular-grid neighbors are not.
        sum = to_log(own.rgb);
        moment = sum * sum;
        weight_sum = 1.0f;
        const float d = hit.read(pixel).z;
        closest = d >= 0.0f ? d : 1e10f;
    }
    sum /= weight_sum;
    moment /= weight_sum;
    const float variance = spatial_variance(sum, moment);
    const float3 forward = transpose(view.ViewRotation.Unpack()) * float3(0, 0, -1);
    const float3 away = view.ScreenPixelScale < 0.0f ? forward : normalize(world - float3(view.CameraPosition));
    const float3 virtual_point = world + away * (closest / max(dot(away, forward), 1e-6f));
    const float4 clip = view.ViewProj.Unpack() * float4(virtual_point, 1);
    output.write(float4(safe_color(from_log(sum)), 0), pixel);
    info.write(float4(variance, clip.z / clip.w, 0, 0), pixel);
}
