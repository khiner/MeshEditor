#ifndef VERTEXWELD_MSL
#define VERTEXWELD_MSL

// Welds vertices identical in every vertex-domain channel and numbers them by first source occurrence.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "gpu/VertexWeldJob.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/TiledJobPushConstants.h"

constant uint WeldEmptySlot = InvalidOffset;

constant uint WeldPositionWords = 3u;
constant uint WeldDeformWords = 8u;
constant uint WeldMorphWords = 6u;
constant uint WeldTangentWords = 3u;

struct WeldContext {
    device const BindlessSet &B;
    constant TiledJobPushConstants &Pc;

    device uint *Storage() const { return BindlessBufferMutable(uint, B.Buffer, Pc.StorageSlot); }
    device const VertexWeldJob *Jobs() const { return reinterpret_cast<device const VertexWeldJob *>(Storage() + Pc.JobsOffset); }
    device const uint2 *Tiles() const { return reinterpret_cast<device const uint2 *>(Storage() + Pc.TileMapOffset); }
    device uint *Scratch() const { return Storage() + Pc.ScratchOffset; }
    device uint *PositionWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.VertexBuffer, job.Positions.Slot) + job.Positions.Offset * WeldPositionWords; }
    device uint *Corners(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.IndexBuffer, job.Corners.Slot) + job.Corners.Offset; }
    device atomic_uint *AtomicScratch() const { return reinterpret_cast<device atomic_uint *>(Scratch()); }
    device uint *DeformWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.BoneDeformBuffer, job.Skin.ValuesSlot); }
    device uint *MorphWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.MorphTargetBuffer, job.Morph.ValuesSlot); }

    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

// Missing channels alias the position buffer because MSL cannot represent an unbound buffer pointer.
struct WeldKeys {
    device uint *Positions;
    device uint *Deform;
    device uint *Morph;
    device const uint *SkinBlocks;
    device const uint *MorphBlocks;
    device uint *Tangents;
    uint Count, TargetCount, FirstVertex;
    bool HasDeform, HasMorph, HasTangents;
};

inline WeldKeys MakeWeldKeys(WeldContext ctx, VertexWeldJob job) {
    device uint *positions = ctx.PositionWords(job);
    const bool has_deform = job.Skin.ValuesSlot != InvalidSlot;
    const bool has_morph = job.Morph.ValuesSlot != InvalidSlot;
    const bool has_tangents = job.TangentOffset != InvalidOffset;
    return {
        .Positions = positions,
        .Deform = has_deform ? ctx.DeformWords(job) : positions,
        .Morph = has_morph ? ctx.MorphWords(job) : positions,
        .SkinBlocks = has_deform ? BindlessBuffer(uint, ctx.B.Buffer, job.Skin.BlocksSlot) : positions,
        .MorphBlocks = has_morph ? BindlessBuffer(uint, ctx.B.Buffer, job.Morph.BlocksSlot) : positions,
        .Tangents = has_tangents ? ctx.Scratch() + job.TangentOffset : positions,
        .Count = job.Count,
        .TargetCount = job.TargetCount,
        .FirstVertex = job.Positions.Offset,
        .HasDeform = has_deform,
        .HasMorph = has_morph,
        .HasTangents = has_tangents,
    };
}

inline uint WeldSkinWord(thread const WeldKeys &k, uint i) { return ElementAttributeIndex(k.SkinBlocks, k.FirstVertex + i) * WeldDeformWords; }
inline uint WeldMorphWord(thread const WeldKeys &k, uint i, uint target) { return ElementAttributeIndex(k.MorphBlocks, k.FirstVertex + i, target) * WeldMorphWords; }

inline uint WeldHashWords(uint hash, device const uint *words, uint first, uint count) {
    for (uint w = 0u; w < count; ++w) {
        hash ^= words[first + w];
        hash *= 16777619u;
    }
    return hash;
}

inline uint WeldKeyHash(thread const WeldKeys &k, uint i) {
    uint hash = WeldHashWords(2166136261u, k.Positions, i * WeldPositionWords, WeldPositionWords);
    if (k.HasDeform) hash = WeldHashWords(hash, k.Deform, WeldSkinWord(k,i), WeldDeformWords);
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        hash = WeldHashWords(hash, k.Morph, WeldMorphWord(k,i,t), WeldMorphWords);
    }
    for (uint t = 0u; k.HasTangents && t < k.TargetCount; ++t) {
        hash = WeldHashWords(hash, k.Tangents, (t * k.Count + i) * WeldTangentWords, WeldTangentWords);
    }
    return hash;
}

inline bool WeldWordsEqual(device const uint *words, uint first_a, uint first_b, uint count) {
    for (uint w = 0u; w < count; ++w) {
        if (words[first_a + w] != words[first_b + w]) return false;
    }
    return true;
}

inline bool WeldKeysEqual(thread const WeldKeys &k, uint a, uint b) {
    if (!WeldWordsEqual(k.Positions, a * WeldPositionWords, b * WeldPositionWords, WeldPositionWords)) return false;
    if (k.HasDeform && !WeldWordsEqual(k.Deform, WeldSkinWord(k,a), WeldSkinWord(k,b), WeldDeformWords)) return false;
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        if (!WeldWordsEqual(k.Morph, WeldMorphWord(k,a,t), WeldMorphWord(k,b,t), WeldMorphWords)) return false;
    }
    for (uint t = 0u; k.HasTangents && t < k.TargetCount; ++t) {
        if (!WeldWordsEqual(k.Tangents, (t * k.Count + a) * WeldTangentWords, (t * k.Count + b) * WeldTangentWords, WeldTangentWords)) return false;
    }
    return true;
}

// Copies `count` words between a staged welded record and one channel, then advances the record cursor.
inline void WeldMoveWords(device uint *record, thread uint &w, device uint *channel, uint first, uint count, bool to_channels) {
    for (uint p = 0u; p < count; ++p) {
        if (to_channels) channel[first + p] = record[w + p];
        else record[w + p] = channel[first + p];
    }
    w += count;
}

// Copies one welded vertex's channels.
// `stride` addresses the host tangent scratch.
inline void WeldMoveRecord(thread const WeldKeys &k, device uint *record, uint vertex_index, uint stride, bool to_channels) {
    uint w = 0u;
    WeldMoveWords(record, w, k.Positions, vertex_index * WeldPositionWords, WeldPositionWords, to_channels);
    if (k.HasDeform) WeldMoveWords(record, w, k.Deform, WeldSkinWord(k,vertex_index), WeldDeformWords, to_channels);
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        WeldMoveWords(record, w, k.Morph, WeldMorphWord(k,vertex_index,t), WeldMorphWords, to_channels);
    }
    for (uint t = 0u; k.HasTangents && t < k.TargetCount; ++t) {
        WeldMoveWords(record, w, k.Tangents, (t * stride + vertex_index) * WeldTangentWords, WeldTangentWords, to_channels);
    }
}

kernel void VertexWeldTableInit(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.TableMask) return;
    ctx.Scratch()[job.TableOffset + i] = WeldEmptySlot;
    if (i <= job.Count) ctx.Scratch()[job.FlagsOffset + i] = 0u;
}

kernel void VertexWeldInsert(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.Count) return;
    const WeldKeys keys = MakeWeldKeys(ctx, job);
    device atomic_uint *table = ctx.AtomicScratch() + job.TableOffset;
    // Equal keys share a probe sequence, and atomic min selects their lowest source index.
    uint slot = WeldKeyHash(keys, i) & job.TableMask;
    for (;;) {
        uint occupant = WeldEmptySlot;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &occupant, i, memory_order_relaxed, memory_order_relaxed)) break;
        if (occupant == WeldEmptySlot) continue;
        if (WeldKeysEqual(keys, occupant, i)) {
            atomic_fetch_min_explicit(&table[slot], i, memory_order_relaxed);
            break;
        }
        slot = (slot + 1u) & job.TableMask;
    }
    ctx.Scratch()[job.SlotOffset + i] = slot;
}

kernel void VertexWeldMarkReps(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= (job.KeepLooseVertices ? job.Count : job.CornerCount)) return;
    device uint *scratch = ctx.Scratch();
    // Importers may retain only referenced vertices; both modes use the same channel compaction.
    const uint v = job.KeepLooseVertices ? i : ctx.Corners(job)[i] - job.Positions.Offset;
    const uint representative = scratch[job.TableOffset + scratch[job.SlotOffset + v]];
    atomic_store_explicit(ctx.AtomicScratch() + job.FlagsOffset + representative, 1u, memory_order_relaxed);
}

kernel void VertexWeldBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    ScanBlockSum(
        ctx.Scratch() + job.FlagsOffset, job.Count + 1u, tile.y, ctx.Scratch() + job.BlockOffset,
        lane, simd_lane, simd_group, sums
    );
}

kernel void VertexWeldBlockPrefix(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const WeldContext ctx{bindless, pc};
    const VertexWeldJob job = ctx.Jobs()[group_id];
    ScanBlockPrefix(ctx.Scratch() + job.BlockOffset, job.BlockCount, lane, simd_lane, simd_group, sums);
}

kernel void VertexWeldScan(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device uint *flags = ctx.Scratch() + job.FlagsOffset;
    ScanBlockOffsets(flags, job.Count + 1u, tile.y, ctx.Scratch() + job.BlockOffset, flags, lane, simd_lane, simd_group, sums);
}

kernel void VertexWeldEmit(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.Count) return;
    device uint *scratch = ctx.Scratch();
    device const uint *welded_index = scratch + job.FlagsOffset;
    const uint representative = scratch[job.TableOffset + scratch[job.SlotOffset + i]];
    scratch[job.RemapOffset + i] = welded_index[representative];
    if (representative == i && welded_index[i] != welded_index[i + 1u]) scratch[job.RepsOffset + welded_index[i]] = i;
}

kernel void VertexWeldCompact(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device const uint *scratch = ctx.Scratch();
    const uint welded = scratch[job.FlagsOffset + job.Count];
    // Skip channel compaction when every source vertex is retained.
    if (welded == job.Count) return;
    const uint n = tile.y * ScanTileSize + lane;
    if (n >= welded) return;
    const uint source = scratch[job.RepsOffset + n];
    const WeldKeys keys = MakeWeldKeys(ctx, job);
    WeldMoveRecord(keys, ctx.Scratch() + job.CompactOffset + n * job.RecordWords, source, keys.Count, false);
}

kernel void VertexWeldWriteBack(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device const uint *scratch = ctx.Scratch();
    const uint welded = scratch[job.FlagsOffset + job.Count];
    if (welded == job.Count) return;
    const uint n = tile.y * ScanTileSize + lane;
    if (n >= welded) return;
    // Write canonical channels and repack host tangent deltas with welded-count stride.
    const WeldKeys keys = MakeWeldKeys(ctx, job);
    WeldMoveRecord(keys, ctx.Scratch() + job.CompactOffset + n * job.RecordWords, n, welded, true);
}

kernel void VertexWeldRemapCorners(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    // Skip corner remapping when every source vertex is retained.
    if (ctx.Scratch()[job.FlagsOffset + job.Count] == job.Count) return;
    const uint c = tile.y * ScanTileSize + lane;
    if (c >= job.CornerCount) return;
    device uint *corners = ctx.Corners(job);
    corners[c] = job.Positions.Offset + ctx.Scratch()[job.RemapOffset + corners[c] - job.Positions.Offset];
}

#endif
