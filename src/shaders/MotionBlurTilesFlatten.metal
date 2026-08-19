#ifndef MOTIONBLURTILESFLATTEN_MSL
#define MOTIONBLURTILESFLATTEN_MSL

// One threadgroup per tile, so the threadgroup id is the tile coordinate. Each thread reduces a
// fixed share of the tile's pixels on its own, and only its winner reaches threadgroup memory.
#include "Bindless.metal"
#include "MotionBlurShared.metal"
#include "Velocity.metal"
#include "MotionBlurTilesFlattenPushConstants.metal"

constant int FlattenThreads = 8;
constant int FlattenBlocks = MotionBlurTileSize / FlattenThreads;

// Reduction payload: motion length in the high bits so an atomic max sorts by it, and the pixel's
// place in the tile below. Every pixel gets a distinct payload, so one thread alone holds the tile's
// winner and can broadcast its vector. Equal-length pixels resolve by position, row first.
inline uint PackLocal(float2 motion, uint2 tile_coord) {
    return (min(uint(ceil(length(motion))), 0xFFFFu) << 16u) | (tile_coord.y << 5) | tile_coord.x;
}

// The tile image is written here, so this pass takes the write view of the bindless set.
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
        // Zero this tile's indirection entries: an untouched entry would read as tile (0,0)'s
        // motion. The encoder orders these writes ahead of the dilate pass's atomics.
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

    // Each step reads one thread-sized block of the tile, so neighbouring threads read neighbouring
    // pixels. Pixels past the render edge clamp onto the edge pixel, which a max reduction absorbs.
    for (int i = 0; i < FlattenBlocks * FlattenBlocks; ++i) {
        const int2 block = int2(i % FlattenBlocks, i / FlattenBlocks) * FlattenThreads;
        const uint2 tile_coord = uint2(block) + local_id;
        const int2 texel = min(tile_origin + int2(tile_coord), render_size - 1);
        const float2 uv = (float2(texel) + 0.5f) / float2(render_size);
        float4 motion = UnpackVelocity(scene.FetchTex(pc.VelocitySamplerSlot, texel, 0));

        // Clip to the viewport in NDC so a streak stops at the frame edge. The next-motion is stored
        // pointing backward, so negating it aims the ray the way the motion actually goes.
        float2 line_clip;
        line_clip.x = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, motion.xy * 2.0f);
        line_clip.y = LineUnitSquareIntersectDistSafe(uv * 2.0f - 1.0f, -motion.zw * 2.0f);
        motion *= min(line_clip, float2(1.0f)).xxyy;
        // UV to pixels, then to shutter-relative motion. Past here both halves point forward in time.
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

    // The winning thread publishes its own vector, which avoids an atomic max over floats.
    if (local_payload_prev == atomic_load_explicit(&payload[MotionPrev], memory_order_relaxed)) max_motion[MotionPrev] = local_max_prev;
    if (local_payload_next == atomic_load_explicit(&payload[MotionNext], memory_order_relaxed)) max_motion[MotionNext] = local_max_next;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_index == 0u) {
        bindless.Image[pc.TileImageSlot].write(float4(max_motion[MotionPrev], max_motion[MotionNext]), group_id);
    }
}

#endif
