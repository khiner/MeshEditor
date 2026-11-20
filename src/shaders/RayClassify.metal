#include "RayGuide.h"
#include "RayWork.h"
#include "gpu/RayLightingPushConstants.h"
#include "gpu/SceneViewUBO.h"

kernel void RayClassify(texture2d<float, access::read> guide [[texture(1)]], depth2d<float, access::read> depth [[texture(3)]],
    texture2d<float, access::read> reflection [[texture(10)]], texture2d<float, access::read> diffuse [[texture(11)]],
    texture2d<float, access::write> reflection_history [[texture(4)]], texture2d<float, access::write> reflection_moments [[texture(5)]],
    texture2d<float, access::write> diffuse_history [[texture(6)]], texture2d<float, access::write> diffuse_moments [[texture(7)]], constant RayLightingPushConstants &pc [[buffer(1)]], constant SceneViewUBO &view [[buffer(2)]], device uint *work [[buffer(9)]], uint2 pixel [[thread_position_in_grid]], uint local_index [[thread_index_in_threadgroup]]) {
    const uint2 extent = uint2(guide.get_width(), guide.get_height());
    const uint step = view.RayStep;
    const uint frame = view.RayFrame + pc.SampleOffset;
    const uint2 offset = RayGridOffset(frame, step);
    const uint2 cell = (pixel / step) * step;
    const uint2 center = min(cell + offset, extent - 1u);
    const float4 g = guide.read(pixel), cg = guide.read(center);
    const bool active_pixel = RayGuideActive(g, view.RayFastGI, view.RayReflections, view.RayMaxRoughness);
    const bool active_center = RayGuideActive(cg, view.RayFastGI, view.RayReflections, view.RayMaxRoughness);
    // EEVEE's compact 8x8 denoising tile list shares the native ray work allocation.
    threadgroup atomic_uint active;
    if (local_index == 0u) atomic_store_explicit(&active, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mask = uint(active_pixel);
    if (pc.DiffuseTileOffset && active_pixel && RayGuideDeferred(g)) {
        mask = uint(any(reflection.read(pixel).rgb != 0.0f));
        if (g.w >= 0.0f && any(diffuse.read(pixel).rgb != 0.0f)) mask |= 2u;
    }
    const uint simd_mask = simd_or(mask);
    if (simd_is_first() && simd_mask) atomic_fetch_or_explicit(&active, simd_mask, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tile_mask = atomic_load_explicit(&active, memory_order_relaxed);
    if (local_index == 0u) {
        const uint tile = (pixel.y / 8u) * ((extent.x + 7u) / 8u) + pixel.x / 8u;
        if (tile_mask & 1u) {
            const uint index = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(work + 4), 1u, memory_order_relaxed);
            work[RayTileOffset(extent.x, extent.y) + index] = tile;
        }
        if (tile_mask & 2u) {
            device uint *list = work + pc.DiffuseTileOffset;
            const uint index = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(list), 1u, memory_order_relaxed);
            list[3u + index] = tile;
        }
    }
    // Skipped temporal dispatches must invalidate their ping-pong history.
    if (pc.DiffuseTileOffset) {
        if (!(tile_mask & 1u)) {
            reflection_history.write(float4(0), pixel);
            reflection_moments.write(float4(0), pixel);
        }
        if (!(tile_mask & 2u)) {
            diffuse_history.write(float4(0), pixel);
            diffuse_moments.write(float4(0), pixel);
        }
    }
    bool sample = false;
    if (active_pixel) {
        const float3 normal = DecodeRayNormal(cg.xy);
        sample = (RayGuideRoughness(g) <= 0.08f && !RayGuideGlass(g)) || all(pixel == center) || !active_center || g.w != cg.w || dot(normal, DecodeRayNormal(g.xy)) < 0.98f || abs(RayGuideRoughness(g) - RayGuideRoughness(cg)) > 0.05f;
        if (pc.QuincunxGlass && RayGuideSmoothGlass(g)) {
            // The second sample is half a cell away in each axis.
            const uint2 opposite = (offset + step / 2u) % step;
            sample |= all(pixel == min(cell + opposite, extent - 1u));
        }
        if (g.z >= 4.0f) sample = all(pixel == center) || !active_center || g.w != cg.w;
        if (!sample && g.z < 4.0f) {
            const float4x4 inverse = pc.ClipToWorld.Unpack();
            const float z = depth.read(center);
            const float3 world = RayWorldPoint(float2(center), z, float2(extent), inverse);
            const float width = max(distance(world, RayWorldPoint(float2(center) + float2(1, 0), z, float2(extent), inverse)), 1e-6f);
            const float3 point = RayWorldPoint(float2(pixel), depth.read(pixel), float2(extent), inverse);
            sample = abs(dot(normal, point - world)) > width * 0.5f;
        }
    }
    RayAppendPixel(work, pixel.y * extent.x + pixel.x, sample);
}

kernel void RayDispatch(device uint *work [[buffer(9)]]) {
    work[1] = (work[0] + 63u) / 64u;
    work[2] = work[3] = 1u;
    work[5] = work[4];
    work[6] = work[7] = 1u;
}

kernel void RayReset(device uint *work [[buffer(9)]]) { work[0] = work[4] = 0u; }
