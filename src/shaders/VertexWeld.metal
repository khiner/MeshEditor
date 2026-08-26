#ifndef VERTEXWELD_MSL
#define VERTEXWELD_MSL

// Merges the vertices of a mesh identical in every vertex-domain channel, in ten passes over one
// job list, each pass reading its own tiles at the push constant's FirstTile: clear the table,
// insert every vertex, mark the vertices that represent a key, scan the marks, read out each
// vertex's welded index, then compact the channels and rewrite the corners through it.
// A key's table slot depends on arrival order, but its entry settles at the lowest vertex index
// holding that key, so the welded numbering follows first appearance whatever order the inserts ran in.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "VertexWeldJob.metal"
#include "VertexWeldPushConstants.metal"

constant uint WeldEmptySlot = 0xffffffffu;

// Words each channel takes per vertex: a position, a joint index and weight per bone slot, one
// target's position and normal deltas, and one target's tangent delta.
constant uint WeldPositionWords = 3u;
constant uint WeldDeformWords = 8u;
constant uint WeldMorphWords = 6u;
constant uint WeldTangentWords = 3u;

struct WeldContext {
    device const BindlessSet &B;
    constant VertexWeldPushConstants &Pc;

    device const VertexWeldJob *Jobs() const { return BindlessBuffer(VertexWeldJob, B.Buffer, Pc.JobsSlot); }
    device const uint2 *Tiles() const { return BindlessBuffer(uint2, B.Buffer, Pc.TileMapSlot); }
    device uint *Scratch() const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot); }
    device uint *PositionWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.VertexBuffer, job.Positions.Slot) + job.Positions.Offset * WeldPositionWords; }
    device uint *Corners(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.IndexBuffer, job.Corners.Slot) + job.Corners.Offset; }
    device atomic_uint *AtomicScratch() const { return BindlessBufferMutable(atomic_uint, B.Buffer, Pc.ScratchSlot); }
    device uint *DeformWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.BoneDeformBuffer, job.Deform.Slot) + job.Deform.Offset * WeldDeformWords; }
    device uint *MorphWords(VertexWeldJob job) const { return BindlessBufferMutable(uint, B.MorphTargetBuffer, job.Morph.Slot) + job.Morph.Offset * WeldMorphWords; }

    // The tile this threadgroup runs: its job and the tile's place within that job's pass.
    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

// A channel the mesh lacks aliases the positions and stays unread, since MSL has no null pointer to
// hand a buffer that was never bound.
struct WeldKeys {
    device uint *Positions;
    device uint *Deform;
    device uint *Morph;
    device uint *Tangents;
    uint Count, TargetCount;
    bool HasDeform, HasMorph, HasTangents;
};

inline WeldKeys MakeWeldKeys(WeldContext ctx, VertexWeldJob job) {
    WeldKeys k;
    k.Positions = ctx.PositionWords(job);
    k.Count = job.Count;
    k.TargetCount = job.TargetCount;
    k.HasDeform = job.Deform.Slot != INVALID_SLOT;
    k.HasMorph = job.Morph.Slot != INVALID_SLOT;
    k.HasTangents = job.TangentOffset != INVALID_OFFSET;
    k.Deform = k.HasDeform ? ctx.DeformWords(job) : k.Positions;
    k.Morph = k.HasMorph ? ctx.MorphWords(job) : k.Positions;
    k.Tangents = k.HasTangents ? ctx.Scratch() + job.TangentOffset : k.Positions;
    return k;
}

inline uint WeldHashWords(uint hash, device const uint *words, uint first, uint count) {
    for (uint w = 0u; w < count; ++w) {
        hash ^= words[first + w];
        hash *= 16777619u;
    }
    return hash;
}

inline uint WeldKeyHash(thread const WeldKeys &k, uint i) {
    uint hash = WeldHashWords(2166136261u, k.Positions, i * WeldPositionWords, WeldPositionWords);
    if (k.HasDeform) hash = WeldHashWords(hash, k.Deform, i * WeldDeformWords, WeldDeformWords);
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        hash = WeldHashWords(hash, k.Morph, (t * k.Count + i) * WeldMorphWords, WeldMorphWords);
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
    if (k.HasDeform && !WeldWordsEqual(k.Deform, a * WeldDeformWords, b * WeldDeformWords, WeldDeformWords)) return false;
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        if (!WeldWordsEqual(k.Morph, (t * k.Count + a) * WeldMorphWords, (t * k.Count + b) * WeldMorphWords, WeldMorphWords)) return false;
    }
    for (uint t = 0u; k.HasTangents && t < k.TargetCount; ++t) {
        if (!WeldWordsEqual(k.Tangents, (t * k.Count + a) * WeldTangentWords, (t * k.Count + b) * WeldTangentWords, WeldTangentWords)) return false;
    }
    return true;
}

// Moves `count` words between a welded vertex's record in the compaction staging and one channel,
// advancing the record cursor past them.
inline void WeldMoveWords(device uint *record, thread uint &w, device uint *channel, uint first, uint count, bool to_channels) {
    for (uint p = 0u; p < count; ++p) {
        if (to_channels) channel[first + p] = record[w + p];
        else record[w + p] = channel[first + p];
    }
    w += count;
}

// Moves one welded vertex's channels to or from its record, at `stride` vertices per morph target.
inline void WeldMoveRecord(thread const WeldKeys &k, device uint *record, uint vertex_index, uint stride, bool to_channels) {
    uint w = 0u;
    WeldMoveWords(record, w, k.Positions, vertex_index * WeldPositionWords, WeldPositionWords, to_channels);
    if (k.HasDeform) WeldMoveWords(record, w, k.Deform, vertex_index * WeldDeformWords, WeldDeformWords, to_channels);
    for (uint t = 0u; k.HasMorph && t < k.TargetCount; ++t) {
        WeldMoveWords(record, w, k.Morph, (t * stride + vertex_index) * WeldMorphWords, WeldMorphWords, to_channels);
    }
    for (uint t = 0u; k.HasTangents && t < k.TargetCount; ++t) {
        WeldMoveWords(record, w, k.Tangents, (t * stride + vertex_index) * WeldTangentWords, WeldTangentWords, to_channels);
    }
}

kernel void VertexWeldTableInit(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.TableMask) return;
    ctx.Scratch()[job.TableOffset + i] = WeldEmptySlot;
}

kernel void VertexWeldInsert(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.Count) return;
    const WeldKeys keys = MakeWeldKeys(ctx, job);
    device atomic_uint *table = ctx.AtomicScratch() + job.TableOffset;
    // Equal keys probe the same slots in the same order, so they meet at the first slot that is
    // empty or already theirs, and the lowest index among them wins the entry.
    uint slot = WeldKeyHash(keys, i) & job.TableMask;
    for (;;) {
        uint occupant = WeldEmptySlot;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &occupant, i, memory_order_relaxed, memory_order_relaxed)) break;
        if (occupant == WeldEmptySlot) continue; // The exchange can fail without the slot being taken.
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
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.Count) return;
    device uint *scratch = ctx.Scratch();
    // The terminator carries no mark, so the scan leaves the welded count in its slot.
    const bool represents = i < job.Count && scratch[job.TableOffset + scratch[job.SlotOffset + i]] == i;
    scratch[job.FlagsOffset + i] = represents ? 1u : 0u;
}

kernel void VertexWeldBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
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
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
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
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device uint *flags = ctx.Scratch() + job.FlagsOffset;
    uint local[ScanPerThread];
    uint start = ScanBlockStart(
        flags, job.Count + 1u, tile.y, ctx.Scratch() + job.BlockOffset, lane, simd_lane, simd_group, sums, local
    );
    // Each thread rewrites only the marks it read, so the scan lands in place.
    const uint base = tile.y * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint i = base + k;
        if (i > job.Count) break;
        flags[i] = start;
        start += local[k];
    }
}

kernel void VertexWeldEmit(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
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
    if (representative == i) scratch[job.RepsOffset + welded_index[i]] = i;
}

kernel void VertexWeldCompact(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device const uint *scratch = ctx.Scratch();
    const uint welded = scratch[job.FlagsOffset + job.Count];
    // A mesh that merges nothing represents every vertex by itself, so its channels already stand welded.
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
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    device const uint *scratch = ctx.Scratch();
    const uint welded = scratch[job.FlagsOffset + job.Count];
    if (welded == job.Count) return;
    const uint n = tile.y * ScanTileSize + lane;
    if (n >= welded) return;
    // Each target's deltas restride to the welded count, which is where the shrunk arena reads them.
    const WeldKeys keys = MakeWeldKeys(ctx, job);
    WeldMoveRecord(keys, ctx.Scratch() + job.CompactOffset + n * job.RecordWords, n, welded, true);
}

kernel void VertexWeldRemapCorners(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexWeldPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const WeldContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexWeldJob job = ctx.Jobs()[tile.x];
    // A kept-vertex total equal to the input count means the mesh merged nothing, so its corners already point right.
    if (ctx.Scratch()[job.FlagsOffset + job.Count] == job.Count) return;
    const uint c = tile.y * ScanTileSize + lane;
    if (c >= job.CornerCount) return;
    device uint *corners = ctx.Corners(job);
    corners[c] = ctx.Scratch()[job.RemapOffset + corners[c]];
}

#endif
