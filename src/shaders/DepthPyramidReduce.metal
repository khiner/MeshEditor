#ifndef DEPTHPYRAMIDREDUCE_MSL
#define DEPTHPYRAMIDREDUCE_MSL

// Reduces up to six pyramid levels in one dispatch. Each threadgroup computes a 32x32 tile of the
// first level as the farthest depth of clamped 2x2 source reads, then halves the tile down through
// threadgroup memory, storing every level. Source reads clamp to the valid bounds, so each level's
// last texel absorbs the source edge and every valid source texel is covered by some destination texel.
#include "Bindless.metal"
#include "DepthPyramidReducePushConstants.metal"

constant uint INVALID_SLOT_PYRAMID = 0xffffffffu;
constant int PyramidTileDim = 32;

inline float max4(float a, float b, float c, float d) { return max(max(a, b), max(c, d)); }

// The pyramid levels are written here, so this pass takes the write view of the bindless set.
kernel void DepthPyramidReduceKernel(
    uint2 local_id [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    threadgroup float *tile [[threadgroup(0)]],
    device const BindlessSetImageWrite &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant DepthPyramidReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const SceneImageWrite scene{bindless, view, theme, workspace};
    const int2 thread_px = int2(local_id);
    const int2 tile_base = int2(group_id) * PyramidTileDim;
    const int2 src_max = int2(int(pc.SrcWidth), int(pc.SrcHeight)) - 1;
    const int2 dst_size = int2(bindless.Image[pc.DstSlots[0]].get_width(), bindless.Image[pc.DstSlots[0]].get_height());
    const uint lod = pc.SrcLod;
    // First level: each thread reduces a 2x2 block of destination texels.
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            const int2 local = thread_px * 2 + int2(i, j);
            const int2 dst = tile_base + local;
            const int2 src = dst * 2;
            const float v = max4(
                scene.FetchTex(pc.SrcSamplerSlot, min(src, src_max), lod).r,
                scene.FetchTex(pc.SrcSamplerSlot, min(src + int2(1, 0), src_max), lod).r,
                scene.FetchTex(pc.SrcSamplerSlot, min(src + int2(0, 1), src_max), lod).r,
                scene.FetchTex(pc.SrcSamplerSlot, min(src + int2(1, 1), src_max), lod).r
            );
            if (all(dst < dst_size)) bindless.Image[pc.DstSlots[0]].write(float4(v), uint2(dst));
            tile[local.y * PyramidTileDim + local.x] = v;
        }
    }
    for (int level = 1; level <= 5; ++level) {
        const uint slot = pc.DstSlots[level];
        if (slot == INVALID_SLOT_PYRAMID) return; // Uniform across the threadgroup: the chain ended.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int dim = PyramidTileDim >> level;
        const bool reducing = thread_px.x < dim && thread_px.y < dim;
        float v = 0.0f;
        if (reducing) {
            const int2 s = thread_px * 2;
            v = max4(
                tile[s.y * PyramidTileDim + s.x], tile[s.y * PyramidTileDim + s.x + 1],
                tile[(s.y + 1) * PyramidTileDim + s.x], tile[(s.y + 1) * PyramidTileDim + s.x + 1]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (reducing) {
            tile[thread_px.y * PyramidTileDim + thread_px.x] = v;
            const int2 dst = (tile_base >> level) + thread_px;
            if (all(dst < max(dst_size >> level, int2(1)))) bindless.Image[slot].write(float4(v), uint2(dst));
        }
    }
}

#endif
