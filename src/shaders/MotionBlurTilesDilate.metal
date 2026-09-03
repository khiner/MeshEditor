#ifndef MOTIONBLURTILESDILATE_MSL
#define MOTIONBLURTILESDILATE_MSL

#include "Bindless.metal"
#include "MotionBlurShared.metal"
#include "MotionBlurTilesDilatePushConstants.metal"

// A tile's own bounding circle plus the line's, both of radius sqrt(1/2), sum to sqrt(2).
constant float TileCoverRadius = 1.41421356237309504880f;

inline float2 SafeNormalize(float2 v) {
    const float len_sq = dot(v, v);
    return len_sq > 1e-35f ? v * rsqrt(len_sq) : float2(1.0f, 0.0f);
}

inline bool IsInsideMotionLine(int2 tile, float2 origin, float2 normal) {
    return abs(dot(normal, origin - float2(tile))) < TileCoverRadius;
}

kernel void MotionBlurTilesDilateKernel(
    uint2 global_id [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MotionBlurTilesDilatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const int2 src_tile = int2(global_id);
    const int2 tile_extent = int2(bindless.Image[pc.TileImageSlot].get_width(), bindless.Image[pc.TileImageSlot].get_height());
    if (any(src_tile >= tile_extent)) return;

    device atomic_uint *indirections = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.TileIndirectionSlot);
    const float4 max_motion = bindless.Image[pc.TileImageSlot].read(uint2(src_tile));
    const uint payload_prev = MotionTilePack(max_motion.xy, uint2(src_tile));
    const uint payload_next = MotionTilePack(max_motion.zw, uint2(src_tile));

    // Conservatively mark every tile intersected by either motion half for both gather directions.
    for (int half_i = 0; half_i < 2; ++half_i) {
        const float2 motion = half_i == 0 ? max_motion.xy : max_motion.zw;
        const int2 far_tile = src_tile + int2(sign(motion) * ceil(abs(motion) / float(MotionBlurTileSize)));
        const int2 min_tile = max(min(far_tile, src_tile), int2(0));
        const int2 max_tile = min(max(far_tile, src_tile), tile_extent - 1);

        const float2 dir = SafeNormalize(motion);
        const float2 origin = float2(src_tile);
        const float2 normal = float2(-dir.y, dir.x);

        for (int x = min_tile.x; x <= max_tile.x; ++x) {
            for (int y = min_tile.y; y <= max_tile.y; ++y) {
                const int2 tile = int2(x, y);
                if (!IsInsideMotionLine(tile, origin, normal)) continue;
                atomic_fetch_max_explicit(&indirections[MotionTileIndex(MotionPrev, uint2(tile), uint2(tile_extent))], payload_prev, memory_order_relaxed);
                atomic_fetch_max_explicit(&indirections[MotionTileIndex(MotionNext, uint2(tile), uint2(tile_extent))], payload_next, memory_order_relaxed);
            }
        }
    }
}

#endif
