/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal integration of Blender 222853b0d7e eevee_fast_gi.bsl.hh.
 */
#include "EeveeDenoise.h"
#include "EeveeFastGI.h"
#include "RayHistory.h"
#include "gpu/FastGiPushConstants.h"
#include "ibl.metal"
using namespace eevee_fast_gi;
inline uint2 gi_pixel(uint2 p, constant FastGiPushConstants &pc, uint2 size) {
    return RayGridCenter(p, pc.Frame, pc.Step, size);
}
inline float gi_reflection(float roughness, constant SceneViewUBO &view) {
    return view.RayReflections ? RayReflectionWeight(roughness, view.RayMaxRoughness) : 0.f;
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiDepth(depth2d<float, access::read> input [[texture(0)]], texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    output.write(float4(input.read(p)), p);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiDepthReduce(texture2d<float, access::read> input [[texture(0)]], texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    uint2 q = p * 2, size = uint2(input.get_width(), input.get_height());
    float value = 1;
    const uint2 end = select(q + 2, size, p + 1 == uint2(output.get_width(), output.get_height()));
    for (uint y = q.y; y < end.y; ++y)
        for (uint x = q.x; x < end.x; ++x) value = min(value, input.read(uint2(x, y)).r);
    output.write(float4(value), p);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiSetup(texture2d<float, access::read> guide [[texture(0)]], texture2d<float, access::read> current [[texture(1)]], texture2d<float, access::read> geometry [[texture(2)]], texture2d<float, access::read> feedback [[texture(3)]], texture2d<float, access::read> previous_geometry [[texture(4)]], texture2d<float, access::read> previous_normal [[texture(5)]], texture2d<uint, access::read> previous_material [[texture(6)]], texture2d<float, access::read> diffuse [[texture(7)]], texture2d<float, access::read> reflection [[texture(8)]], texture2d<float, access::write> radiance [[texture(9)]], texture2d<float, access::write> normal [[texture(10)]], texture2d<float, access::read> closure_normal [[texture(11)]], device const RayMotionTransform *motion [[buffer(10)]], device const RayDeformedSurface *deformation [[buffer(11)]], constant FastGiPushConstants &pc [[buffer(1)]], device const BindlessSet &bindless [[buffer(0)]], constant SceneViewUBO &view [[buffer(2)]], constant ViewportThemeUbo &theme [[buffer(3)]], constant WorkspaceLights &workspace [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    const uint2 size(guide.get_width(), guide.get_height()), pixel = gi_pixel(p, pc, size);
    const float4 g = guide.read(pixel), stored = RayHistoryGeometry(geometry, pixel, pc.ClipToWorld.Unpack());
    const uint tag = as_type<uint>(stored.w);
    const bool valid = (tag & 0x7fffffffu) != 0 && g.w >= 0;
    const float3 geometric = DecodeRayNormal(g.xy);
    const float3 n = g.z >= 4.f ? DecodeRayNormal(closure_normal.read(pixel).xy) : geometric;
    const float3 vn = (pc.WorldToView.Unpack() * float4(n, 0)).xyz;
    float3 light = current.read(pixel).rgb, old;
    const Scene scene{bindless, view, theme, workspace};
    if (g.z >= 4.f) light += getDiffuseLight(scene, n) * view.Exposure * (diffuse.read(pixel).rgb + reflection.read(pixel).rgb * (1.f - gi_reflection(g.z - 4.f, view)));
    float3 previous_position = stored.xyz, previous_geometric = geometric;
    if (valid && pc.HistoryValid && (tag & 0x80000000u) == 0 && RayPreviousPixelSurface(motion, (tag & 0x7fffffffu) - 1, pc.ApplyMotion != 0, deformation, pixel.y * size.x + pixel.x, previous_position, previous_geometric) && RayFeedback(feedback, previous_geometry, previous_normal, previous_material, motion, pc.PreviousViewProj.Unpack(), pc.PreviousClipToWorld.Unpack(), false, previous_position, previous_geometric, (tag & 0x7fffffffu) - 1, uint(g.w), old)) light = old;
    radiance.write(float4(valid ? eevee_denoise::safe_color(light) : float3(0), 0), p);
    normal.write(float4(vn, valid && g.z >= 4.f ? 1.f : 0.f), p);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiReduce(texture2d<float, access::read> input_radiance [[texture(0)]], texture2d<float, access::read> input_normal [[texture(1)]], texture2d<float, access::write> radiance [[texture(2)]], texture2d<float, access::write> normal [[texture(3)]], uint2 p [[thread_position_in_grid]]) {
    const uint2 size(input_radiance.get_width(), input_radiance.get_height());
    float4 r(0), n(0);
    for (uint y = 0; y < 2; ++y)
        for (uint x = 0; x < 2; ++x) {
            const uint2 q = min(p * 2 + uint2(x, y), size - 1);
            r += input_radiance.read(q);
            n += input_normal.read(q);
        }
    radiance.write(r * .25f, p);
    normal.write(n * .25f, p);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiScan(texture2d<float> depth [[texture(0)]], texture2d<float> radiance [[texture(1)]], texture2d<float> normals [[texture(2)]], texture2d<float, access::read> guide [[texture(3)]], texture2d_array<float, access::write> output [[texture(4)]], texture2d_array<float> lut [[texture(15)]], constant FastGiPushConstants &pc [[buffer(1)]], constant SceneViewUBO &view [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    const uint2 size(guide.get_width(), guide.get_height()), pixel = gi_pixel(p, pc, size);
    SH result{};
    if (guide.read(pixel).z < 4.f) {
        store(output, p, result);
        return;
    }
    constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
    constexpr sampler linear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    const float2 uv = (float2(pixel) + .5f) / float2(size);
    const float4x4 inverse = pc.InverseProjection.Unpack(), projection = pc.Projection.Unpack();
    const float3 P = view_point(uv, depth.sample(nearest, uv, level(0)).r, inverse), N = normals.read(p).xyz;
    const float3 V = projection[3][3] == 0 ? normalize(-P) : float3(0, 0, 1);
    const float4 noise = fract(lut.read(p % 64u, 0u) + float4(pc.Frame) * float4(.754877666f, .569840296f, .438579f, .754877666f));
    const float phi = noise.x * M_PI_F * .5f;
    float2 direction(cos(phi), sin(phi));
    float weight_sum = 0;
    constexpr float angle_bias = 1.f / .95f;
#pragma clang loop unroll(full)
    for (uint slice = 0; slice < 2; ++slice) {
        const float3 B = normalize(cross(V, float3(direction, 0))), T = cross(B, V);
        float normal_length, normal_angle;
        projected_normal(N, V, T, B, normal_length, normal_angle);
        normal_angle += (noise.z - .5f) * (M_PI_F / 32.f) * angle_bias;
        SH sh{};
        sh.encode(N, float4(0, 0, 0, 1));
        uint mask = 0;
        for (uint side = 0; side < 2; ++side) {
            const ScreenRay ray = screen_ray(P, float3(side == 0 ? direction : -direction, 0), view.RayGiDistance, projection, float2(size));
            ThicknessEstimator estimator{float2(0), P.z, 0};
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const float t = 1.f + pow((float(j) + noise.y) / 8.f, 2.f) * ray.Length;
                const float2 sample_uv = ray.Origin + ray.Direction * t;
                if (any(sample_uv < 0) || any(sample_uv >= 1)) continue;
                const float lod = log2(t * .15f), z = depth.sample(nearest, sample_uv, level(max(0.f, floor(noise.w + lod)))).r;
                if (z >= 1.f) continue;
                const float view_z = view_point(sample_uv, z, inverse).z;
                const float thickness = estimator.thickness(z, t, pixel_thickness(view_z, projection, pc.PixelRadius, view.RayGiThickness));
                const float3 front = view_point(sample_uv, min(z + 4.8e-7f, 1.f), inverse), back = view_point(sample_uv, min(z + thickness, 1.f - 1e-7f), inverse);
                float3 a = (front - P) / view.RayGiDistance, b = (back - P) / view.RayGiDistance;
                const float limit = sqrt(saturate(1.f - dot(a.xy, a.xy)));
                a.z = clamp(a.z, -limit, limit);
                b.z = clamp(b.z, -limit, limit);
                if (a.z == b.z || dot(a, a) < 1e-16f || dot(b, b) < 1e-16f) continue;
                const float3 L = normalize(a);
                float2 theta = acos(clamp(float2(dot(L, V), dot(normalize(b), V)), -1.f, 1.f));
                theta.y = max(theta.x, theta.y);
                if (side) theta = -theta.yx;
                const uint bits = angles_to_bitmask((theta - normal_angle) * angle_bias);
                const float weight = bitmask_to_visibility_uniform(bits & ~mask);
                mask |= bits;
                // Occluded sectors contribute no radiance or SH visibility. Keep the thickness update above.
                if (weight == 0.f) continue;
                const float lod_data = max(0.f, lod - log2(float(pc.Step)));
                float3 light = radiance.sample(linear, sample_uv, level(lod_data)).rgb;
                const float facing = dot(normals.sample(linear, sample_uv, level(lod_data)).xyz, -L);
                if (facing < 0) light *= .25f;
                sh.encode(L, float4(light * (abs(facing) * weight), -weight));
            }
        }
        result.add(sh, normal_length);
        weight_sum += normal_length;
        direction = float2(-direction.y, direction.x);
    }
    const float sphere = 4.f * M_PI_F / max(weight_sum, 1e-8f);
    result.scale(float4(float3(sphere / (1.5f - cos(angle_bias * M_PI_F * .5f))), sphere));
    store(output, p, result);
}
inline float gi_weight(float3 N, float3 P, float3 n, float3 p, float scale) {
    const float d = dot(N, p - P), nd = dot(N, n);
    const float nd2 = nd * nd, nd4 = nd2 * nd2;
    return exp2(-scale * d * d) * saturate(nd4 * nd4);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiDenoise(texture2d_array<float, access::read> input [[texture(0)]], texture2d_array<float, access::write> output [[texture(1)]], texture2d<float, access::read> normal [[texture(2)]], texture2d<float, access::read> depth [[texture(3)]], constant FastGiPushConstants &pc [[buffer(1)]], uint2 p [[thread_position_in_grid]]) {
    const int2 extent(input.get_width(), input.get_height());
    const uint2 full(depth.get_width(), depth.get_height()), pixel = gi_pixel(p, pc, full);
    const float3 P = RayWorldPoint(float2(pixel), depth.read(pixel).r, float2(full), pc.ClipToWorld.Unpack());
    const float3x3 rotation = float3x3(pc.WorldToView.Unpack()[0].xyz, pc.WorldToView.Unpack()[1].xyz, pc.WorldToView.Unpack()[2].xyz);
    const float3 N = transpose(rotation) * normal.read(p).xyz;
    SH sum{};
    float weight_sum = 0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            const int2 q = int2(p) + int2(x, y);
            if (any(q < 0) || any(q >= extent)) continue;
            const float4 n = normal.read(uint2(q));
            if (n.w == 0) continue;
            const uint2 qp = gi_pixel(uint2(q), pc, full);
            const float3 Q = RayWorldPoint(float2(qp), depth.read(qp).r, float2(full), pc.ClipToWorld.Unpack());
            const float weight = max(1e-6f, gi_weight(N, P, transpose(rotation) * n.xyz, Q, 100.f) * exp2(-.9617967f * float(x * x + y * y)));
            sum.add(load(input, uint2(q)), weight);
            weight_sum += weight;
        }
    sum.scale(float4(1.f / max(weight_sum, 1e-8f)));
    store(output, p, sum);
}
[[max_total_threads_per_threadgroup(64)]] kernel void FastGiResolve(texture2d_array<float, access::read> input [[texture(0)]], texture2d<float, access::read> normal [[texture(1)]], texture2d<float, access::read> depth [[texture(2)]], texture2d<float, access::read> guide [[texture(3)]], texture2d<float, access::read> diffuse [[texture(4)]], texture2d<float, access::read> reflection [[texture(5)]], texture2d<float, access::read> closure_normal [[texture(6)]], texture2d<float, access::read_write> color [[texture(7)]], constant FastGiPushConstants &pc [[buffer(1)]], device const BindlessSet &bindless [[buffer(0)]], constant SceneViewUBO &view [[buffer(2)]], constant ViewportThemeUbo &theme [[buffer(3)]], constant WorkspaceLights &workspace [[buffer(4)]], uint2 pixel [[thread_position_in_grid]]) {
    const float4 g = guide.read(pixel);
    if (g.z < 4.f) return;
    const uint2 size(guide.get_width(), guide.get_height());
    const int2 extent(input.get_width(), input.get_height());
    const float3 P = RayWorldPoint(float2(pixel), depth.read(pixel).r, float2(size), pc.ClipToWorld.Unpack()), N = DecodeRayNormal(closure_normal.read(pixel).xy);
    const float3x3 rotation = float3x3(pc.WorldToView.Unpack()[0].xyz, pc.WorldToView.Unpack()[1].xyz, pc.WorldToView.Unpack()[2].xyz);
    const float2 bias = float2(RayGridOffset(pc.Frame, pc.Step)), at = (float2(pixel) - bias) / float(pc.Step);
    const int2 base = int2(floor(at));
    const float2 f = fract(at);
    SH sum{};
    float weight_sum = 0;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            const int2 q = base + int2(x, y);
            if (any(q < 0) || any(q >= extent)) continue;
            const float4 n = normal.read(uint2(q));
            if (n.w == 0) continue;
            const uint2 qp = gi_pixel(uint2(q), pc, size);
            const float3 Q = RayWorldPoint(float2(qp), depth.read(qp).r, float2(size), pc.ClipToWorld.Unpack());
            const float weight = max(1e-4f, gi_weight(N, P, transpose(rotation) * n.xyz, Q, 10000.f)) * (x ? f.x : 1.f - f.x) * (y ? f.y : 1.f - f.y);
            sum.add(load(input, uint2(q)), weight);
            weight_sum += weight;
        }
    const float3 shading = N, V = pc.Projection.Unpack()[3][3] == 0 ? normalize(float3(view.CameraPosition) - P) : transpose(rotation) * float3(0, 0, 1);
    const Scene scene{bindless, view, theme, workspace};
    float3 extra(0);
    for (uint lobe = 0; lobe < 2; ++lobe) {
        const float3 direction = lobe ? dominant_reflection(shading, V, g.z - 4.f) : shading;
        float4 incoming(0, 0, 0, 3);
        if (weight_sum > 0) { incoming = sum.lambert(rotation * direction) / weight_sum; }
        const float3 light = max(incoming.rgb, 0.f) + saturate(incoming.w / 3.f) * getDiffuseLight(scene, direction) * view.Exposure;
        extra += light * (lobe ? reflection.read(pixel).rgb * (1.f - gi_reflection(g.z - 4.f, view)) : diffuse.read(pixel).rgb);
    }
    const float4 old = color.read(pixel);
    color.write(float4(eevee_denoise::safe_color(old.rgb + extra), old.a), pixel);
}
