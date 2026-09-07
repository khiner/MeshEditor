#ifndef MOTIONBLURTILESFLATTEN_MSL
#define MOTIONBLURTILESFLATTEN_MSL

// Reduces each tile to its longest motion vector with one threadgroup per tile.
#include "Bindless.metal"
#include "MotionBlurShared.metal"
#include "VisibilityMotion.metal"
#include "MotionBlurTilesFlattenPushConstants.metal"

constant int FlattenThreads = 8;
constant int FlattenBlocks = MotionBlurTileSize / FlattenThreads;

// Packs motion length above pixel position so atomic max resolves equal lengths by row-major position.
inline uint PackLocal(float2 motion, uint2 tile_coord) {
    return (min(uint(ceil(length(motion))), 0xFFFFu) << 16u) | ((tile_coord.y << 5) + tile_coord.x + 1u);
}

kernel void MotionBlurTilesFlattenKernel(
    uint2 local_id [[thread_position_in_threadgroup]],
    uint local_index [[thread_index_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    threadgroup atomic_uint *payload [[threadgroup(0)]],
    threadgroup float2 *max_motion [[threadgroup(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurTilesFlattenPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    constant SceneViewUBO &previous [[buffer(5)]],
    constant SceneViewUBO &next [[buffer(6)]],
    device atomic_uint *indirections [[buffer(7)]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::write> velocity [[texture(2)]],
    texture2d<float, access::write> tiles [[texture(3)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const Scene prev_scene{bindless, previous, theme, workspace};
    const Scene next_scene{bindless, next, theme, workspace};
    const uint2 tile_extent = uint2(tiles.get_width(), tiles.get_height());

    if (local_index == 0u) {
        atomic_store_explicit(&payload[MotionPrev], 0u, memory_order_relaxed);
        atomic_store_explicit(&payload[MotionNext], 0u, memory_order_relaxed);
        max_motion[MotionPrev] = max_motion[MotionNext] = float2(0.0f);
        // Zero indirection entries before the ordered dilate pass so untouched entries do not reference tile (0, 0).
        atomic_store_explicit(&indirections[MotionTileIndex(MotionPrev, group_id, tile_extent)], 0u, memory_order_relaxed);
        atomic_store_explicit(&indirections[MotionTileIndex(MotionNext, group_id, tile_extent)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint local_payload_prev = 0u;
    uint local_payload_next = 0u;
    float2 local_max_prev = float2(0.0f);
    float2 local_max_next = float2(0.0f);

    const int2 render_size = int2(velocity.get_width(), velocity.get_height());
    const int2 tile_origin = int2(group_id) * MotionBlurTileSize;

    // Each pixel owns its velocity write, including partial edge tiles.
    for (int i = 0; i < FlattenBlocks * FlattenBlocks; ++i) {
        const int2 block = int2(i % FlattenBlocks, i / FlattenBlocks) * FlattenThreads;
        const uint2 tile_coord = uint2(block) + local_id;
        const int2 texel = tile_origin + int2(tile_coord);
        if (any(texel >= render_size)) continue;
        const float2 uv = (float2(texel) + 0.5f) / float2(render_size);
        float4 motion = VisibilityMotion(scene, prev_scene, next_scene, pc, visibility.read(uint2(texel)).r, depth.read(uint2(texel)), uv);
        velocity.write(PackVelocity(motion), uint2(texel));

        // Clip motion to the viewport and negate the backward-stored next-motion vector.
        float2 line_clip;
        line_clip.x = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, motion.xy * 2.0f);
        line_clip.y = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, -motion.zw * 2.0f);
        motion *= min(line_clip, float2(1.0f)).xxyy;
        // Convert UV displacement to shutter-relative pixel motion with both halves directed forward in time.
        motion *= float2(render_size).xyxy;
        motion *= float2(1.0f, -1.0f).xxyy;

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
    if (local_payload_prev != 0u && local_payload_prev == atomic_load_explicit(&payload[MotionPrev], memory_order_relaxed)) max_motion[MotionPrev] = local_max_prev;
    if (local_payload_next != 0u && local_payload_next == atomic_load_explicit(&payload[MotionNext], memory_order_relaxed)) max_motion[MotionNext] = local_max_next;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_index == 0u) {
        tiles.write(float4(max_motion[MotionPrev], max_motion[MotionNext]), group_id);
    }
}

#endif
