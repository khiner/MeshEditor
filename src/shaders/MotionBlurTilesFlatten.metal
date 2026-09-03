#ifndef MOTIONBLURTILESFLATTEN_MSL
#define MOTIONBLURTILESFLATTEN_MSL

// Reduces each tile to its longest motion vector with one threadgroup per tile.
#include "Bindless.metal"
#include "MotionBlurShared.metal"
#include "Velocity.metal"
#include "MotionBlurTilesFlattenPushConstants.metal"

constant int FlattenThreads = 8;
constant int FlattenBlocks = MotionBlurTileSize / FlattenThreads;

// Packs motion length above pixel position so atomic max resolves equal lengths by row-major position.
inline uint PackLocal(float2 motion, uint2 tile_coord) {
    return (min(uint(ceil(length(motion))), 0xFFFFu) << 16u) | (tile_coord.y << 5) | tile_coord.x;
}

kernel void MotionBlurTilesFlattenKernel(
    uint2 local_id [[thread_position_in_threadgroup]],
    uint local_index [[thread_index_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    threadgroup atomic_uint *payload [[threadgroup(0)]],
    threadgroup float2 *max_motion [[threadgroup(1)]],
    device const BindlessSetImageWrite &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurTilesFlattenPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const SceneImageWrite scene{bindless, view, theme, workspace};
    device atomic_uint *indirections = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.TileIndirectionSlot);
    const uint2 tile_extent = uint2(bindless.Image[pc.TileImageSlot].get_width(), bindless.Image[pc.TileImageSlot].get_height());

    if (local_index == 0u) {
        atomic_store_explicit(&payload[MotionPrev], 0u, memory_order_relaxed);
        atomic_store_explicit(&payload[MotionNext], 0u, memory_order_relaxed);
        // Zero indirection entries before the ordered dilate pass so untouched entries do not reference tile (0, 0).
        atomic_store_explicit(&indirections[MotionTileIndex(MotionPrev, group_id, tile_extent)], 0u, memory_order_relaxed);
        atomic_store_explicit(&indirections[MotionTileIndex(MotionNext, group_id, tile_extent)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint local_payload_prev = 0u;
    uint local_payload_next = 0u;
    float2 local_max_prev = float2(0.0f);
    float2 local_max_next = float2(0.0f);

    const int2 render_size = int2(scene.TexSize(pc.VelocitySamplerSlot, 0));
    const int2 tile_origin = int2(group_id) * MotionBlurTileSize;

    // Clamp partial edge tiles to the final pixel; duplicate values do not affect max reduction.
    for (int i = 0; i < FlattenBlocks * FlattenBlocks; ++i) {
        const int2 block = int2(i % FlattenBlocks, i / FlattenBlocks) * FlattenThreads;
        const uint2 tile_coord = uint2(block) + local_id;
        const int2 texel = min(tile_origin + int2(tile_coord), render_size - 1);
        const float2 uv = (float2(texel) + 0.5f) / float2(render_size);
        float4 motion = UnpackVelocity(scene.FetchTex(pc.VelocitySamplerSlot, texel, 0));

        // Clip motion to the viewport and negate the backward-stored next-motion vector.
        float2 line_clip;
        line_clip.x = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, motion.xy * 2.0f);
        line_clip.y = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, -motion.zw * 2.0f);
        motion *= min(line_clip, float2(1.0f)).xxyy;
        // Convert UV displacement to shutter-relative pixel motion with both halves directed forward in time.
        motion *= float2(render_size).xyxy;
        motion *= float2(pc.MotionScale).xxyy;

        const uint sample_prev = PackLocal(motion.xy, tile_coord);
        if (local_payload_prev < sample_prev) {
            local_payload_prev = sample_prev;
            local_max_prev = motion.xy;
        }
        const uint sample_next = PackLocal(motion.zw, tile_coord);
        if (local_payload_next < sample_next) {
            local_payload_next = sample_next;
            local_max_next = motion.zw;
        }
    }

    atomic_fetch_max_explicit(&payload[MotionPrev], local_payload_prev, memory_order_relaxed);
    atomic_fetch_max_explicit(&payload[MotionNext], local_payload_next, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Publish the winning thread's vector without a float atomic.
    if (local_payload_prev == atomic_load_explicit(&payload[MotionPrev], memory_order_relaxed)) max_motion[MotionPrev] = local_max_prev;
    if (local_payload_next == atomic_load_explicit(&payload[MotionNext], memory_order_relaxed)) max_motion[MotionNext] = local_max_next;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_index == 0u) {
        bindless.Image[pc.TileImageSlot].write(float4(max_motion[MotionPrev], max_motion[MotionNext]), group_id);
    }
}

#endif
