#ifndef VERTEXADJACENCY_MSL
#define VERTEXADJACENCY_MSL

// Builds deterministic vertex-fan or vertex-edge CSR incidence tables matching CPU halfedge order.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "FanItemEncoding.metal"
#include "VertexAdjacencyJob.metal"
#include "VertexAdjacencyKind.metal"
#include "VertexAdjacencyPushConstants.metal"

struct AdjacencyContext {
    device const BindlessSet &B;
    constant VertexAdjacencyPushConstants &Pc;

    device const VertexAdjacencyJob *Jobs() const { return BindlessBuffer(VertexAdjacencyJob, B.Buffer, Pc.JobsSlot); }
    device const uint2 *Tiles() const { return BindlessBuffer(uint2, B.Buffer, Pc.TileMapSlot); }
    device uint *Scratch() const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot); }
    device atomic_uint *AtomicScratch() const { return BindlessBufferMutable(atomic_uint, B.Buffer, Pc.ScratchSlot); }
    device uint *Csr() const { return BindlessBufferMutable(uint, B.Buffer, Pc.AdjacencySlot); }
    device const uint *Corners(VertexAdjacencyJob job) const { return BindlessBuffer(uint, B.IndexBuffer, job.Corners.Slot) + job.Corners.Offset; }

    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

// The halfedge before `h` in its triangle's loop, whose corner vertex is `h`'s from-vertex.
inline uint PreviousHalfedge(uint h) {
    const uint first = h - h % 3u;
    return first + (h - first + 2u) % 3u;
}

// Returns true when `h` is the first halfedge counted by the edge ranks.
inline bool IsEdgeFirst(device const uint *bits, uint h) { return (bits[h / 32u] & (1u << (h % 32u))) != 0u; }

// Returns the number of edge-first bits preceding `h`.
inline uint EdgeIndex(device const uint *bits, device const uint *ranks, uint h) {
    const uint word = h / 32u;
    return ranks[word] + popcount(bits[word] & ((1u << (h % 32u)) - 1u));
}

kernel void VertexAdjacencyZero(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.VertexCount) return;
    ctx.Scratch()[job.CountsOffset + i] = 0u;
}

kernel void VertexAdjacencyCount(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    device const uint *corners = ctx.Corners(job);
    device atomic_uint *counts = ctx.AtomicScratch() + job.CountsOffset;
    if (job.Kind == VertexAdjacencyKind_Fan) {
        atomic_fetch_add_explicit(&counts[corners[h]], 1u, memory_order_relaxed);
        return;
    }
    if (!IsEdgeFirst(ctx.Scratch() + job.EdgeFirstBitsOffset, h)) return;
    atomic_fetch_add_explicit(&counts[corners[PreviousHalfedge(h)]], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[corners[h]], 1u, memory_order_relaxed);
}

kernel void VertexAdjacencyBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    ScanBlockSum(
        ctx.Scratch() + job.CountsOffset, job.VertexCount + 1u, tile.y, ctx.Scratch() + job.BlockOffset,
        lane, simd_lane, simd_group, sums
    );
}

kernel void VertexAdjacencyBlockPrefix(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const VertexAdjacencyJob job = ctx.Jobs()[group_id];
    ScanBlockPrefix(ctx.Scratch() + job.BlockOffset, job.BlockCount, lane, simd_lane, simd_group, sums);
}

kernel void VertexAdjacencyOffsets(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    device uint *counts = ctx.Scratch() + job.CountsOffset;
    uint local[ScanPerThread];
    uint start = ScanBlockStart(
        counts, job.VertexCount + 1u, tile.y, ctx.Scratch() + job.BlockOffset, lane, simd_lane, simd_group, sums, local
    );
    device uint *offsets = ctx.Csr() + job.CsrOffset;
    const uint base = tile.y * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint i = base + k;
        if (i > job.VertexCount) break;
        // Preserve counts as scatter cursors after writing the immutable CSR offsets.
        offsets[i] = start;
        counts[i] = start;
        start += local[k];
    }
}

kernel void VertexAdjacencyScatter(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    device const uint *corners = ctx.Corners(job);
    device atomic_uint *cursors = ctx.AtomicScratch() + job.CountsOffset;
    device uint *items = ctx.Csr() + job.CsrOffset + job.VertexCount + 1u;
    if (job.Kind == VertexAdjacencyKind_Fan) {
        items[atomic_fetch_add_explicit(&cursors[corners[h]], 1u, memory_order_relaxed)] = h;
        return;
    }
    device const uint *bits = ctx.Scratch() + job.EdgeFirstBitsOffset;
    if (!IsEdgeFirst(bits, h)) return;
    const uint edge = EdgeIndex(bits, ctx.Scratch() + job.EdgeFirstRanksOffset, h);
    items[atomic_fetch_add_explicit(&cursors[corners[PreviousHalfedge(h)]], 1u, memory_order_relaxed)] = edge;
    items[atomic_fetch_add_explicit(&cursors[corners[h]], 1u, memory_order_relaxed)] = edge;
}

kernel void VertexAdjacencySort(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexAdjacencyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const AdjacencyContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const VertexAdjacencyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.VertexCount) return;
    device const uint *offsets = ctx.Csr() + job.CsrOffset;
    device uint *items = ctx.Csr() + job.CsrOffset + job.VertexCount + 1u;
    const uint start = offsets[v], end = offsets[v + 1u];
    // Sort each vertex range into CPU emission order; vertex valence bounds the in-thread sort.
    for (uint i = start + 1u; i < end; ++i) {
        const uint key = items[i];
        uint j = i;
        for (; j > start && items[j - 1u] > key; --j) items[j] = items[j - 1u];
        items[j] = key;
    }
    if (job.Kind != VertexAdjacencyKind_Fan) return;
    for (uint i = start; i < end; ++i) {
        const uint h = items[i];
        items[i] = (h / 3u) | ((h % 3u) << FanItemEncoding_LoopShift);
    }
}

#endif
