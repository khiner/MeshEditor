#ifndef MESHCONNECTIVITY_MSL
#define MESHCONNECTIVITY_MSL

// Builds vertex-outgoing halfedges, opposites, and ranked edge-first bits for one triangle mesh.
// Pairing uses the first reverse halfedge in ascending index order to match the CPU store.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "MeshConnectivityJob.metal"
#include "MeshConnectivityPushConstants.metal"

constant uint ConnNullHalfedge = 0xffffffffu;
// Packs the non-bucket endpoint with a high-to-low direction bit.
constant uint ConnReverseBit = 1u << 31;

struct ConnContext {
    device const BindlessSet &B;
    constant MeshConnectivityPushConstants &Pc;

    device const MeshConnectivityJob *Jobs() const { return BindlessBuffer(MeshConnectivityJob, B.Buffer, Pc.JobsSlot); }
    device const uint2 *Tiles() const { return BindlessBuffer(uint2, B.Buffer, Pc.TileMapSlot); }
    device uint *Scratch() const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot); }
    device atomic_uint *AtomicScratch() const { return BindlessBufferMutable(atomic_uint, B.Buffer, Pc.ScratchSlot); }
    device const uint *Corners(MeshConnectivityJob job) const { return BindlessBuffer(uint, B.IndexBuffer, job.Corners.Slot) + job.Corners.Offset; }
    device uint *Run(MeshConnectivityJob job) const { return BindlessBufferMutable(uint, B.Buffer, job.Connectivity.Slot) + job.Connectivity.Offset; }
    device uint *Outgoing(MeshConnectivityJob job) const { return Run(job); }
    device atomic_uint *AtomicOutgoing(MeshConnectivityJob job) const {
        return BindlessBufferMutable(atomic_uint, B.Buffer, job.Connectivity.Slot) + job.Connectivity.Offset;
    }
    device uint *Opposites(MeshConnectivityJob job) const { return Run(job) + job.VertexCount; }
    device uint *EdgeFirstBits(MeshConnectivityJob job) const { return Run(job) + job.VertexCount + job.HalfedgeCount; }
    device uint *EdgeFirstRanks(MeshConnectivityJob job) const { return EdgeFirstBits(job) + job.WordCount; }
    device uint *EdgeSamples(MeshConnectivityJob job) const { return EdgeFirstBits(job) + 2u * job.WordCount; }

    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

// The halfedge before `h` in its triangle's loop, whose corner is `h`'s from-vertex.
inline uint ConnPrevious(uint h) {
    const uint first = h - h % 3u;
    return first + (h - first + 2u) % 3u;
}

inline uint2 ConnEndpoints(device const uint *corners, uint h) { return uint2(corners[ConnPrevious(h)], corners[h]); }

// The key a bucket scan pairs on: the endpoint away from the bucket, marked with the direction.
inline uint ConnBucketKey(device const uint *corners, uint h) {
    const uint2 ends = ConnEndpoints(corners, h);
    return max(ends.x, ends.y) | (ends.x > ends.y ? ConnReverseBit : 0u);
}

kernel void MeshConnectivityZero(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.VertexCount) return;
    ctx.Scratch()[job.CountsOffset + i] = 0u;
    if (i < job.VertexCount) ctx.Outgoing(job)[i] = ConnNullHalfedge;
    if (i == 0u) {
        ctx.Scratch()[job.StateOffset] = 0u;
        ctx.Scratch()[job.StateOffset + 1u] = 0u;
    }
}

kernel void MeshConnectivityCount(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    device const uint *corners = ctx.Corners(job);
    const uint2 ends = ConnEndpoints(corners, h);
    ctx.Opposites(job)[h] = ConnNullHalfedge;
    atomic_fetch_add_explicit(&ctx.AtomicScratch()[job.CountsOffset + min(ends.x, ends.y)], 1u, memory_order_relaxed);
    // Select the lowest outgoing halfedge to match the CPU store.
    atomic_fetch_min_explicit(&ctx.AtomicOutgoing(job)[ends.x], h, memory_order_relaxed);
}

kernel void MeshConnectivityBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    ScanBlockSum(
        ctx.Scratch() + job.CountsOffset, job.VertexCount + 1u, tile.y, ctx.Scratch() + job.BlockOffset,
        lane, simd_lane, simd_group, sums
    );
}

kernel void MeshConnectivityBlockPrefix(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const MeshConnectivityJob job = ctx.Jobs()[group_id];
    ScanBlockPrefix(ctx.Scratch() + job.BlockOffset, job.BlockCount, lane, simd_lane, simd_group, sums);
}

kernel void MeshConnectivityOffsets(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    device uint *counts = ctx.Scratch() + job.CountsOffset;
    device uint *cursors = ctx.Scratch() + job.CursorsOffset;
    uint local[ScanPerThread];
    uint start = ScanBlockStart(
        counts, job.VertexCount + 1u, tile.y, ctx.Scratch() + job.BlockOffset, lane, simd_lane, simd_group, sums, local
    );
    const uint base = tile.y * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint i = base + k;
        if (i > job.VertexCount) break;
        counts[i] = start;
        cursors[i] = start;
        start += local[k];
    }
}

kernel void MeshConnectivityScatter(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    const uint2 ends = ConnEndpoints(ctx.Corners(job), h);
    device atomic_uint *cursors = ctx.AtomicScratch() + job.CursorsOffset;
    const uint slot = atomic_fetch_add_explicit(&cursors[min(ends.x, ends.y)], 1u, memory_order_relaxed);
    ctx.Scratch()[job.ItemsOffset + slot] = h;
}

kernel void MeshConnectivityPair(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.VertexCount) return;
    device uint *scratch = ctx.Scratch();
    device const uint *offsets = scratch + job.CountsOffset;
    device uint *items = scratch + job.ItemsOffset;
    const uint first = offsets[v], last = offsets[v + 1u];
    // Vertex valence bounds the in-thread sort and pairing scan.
    for (uint i = first + 1u; i < last; ++i) {
        const uint key = items[i];
        uint j = i;
        for (; j > first && items[j - 1u] > key; --j) items[j] = items[j - 1u];
        items[j] = key;
    }
    device const uint *corners = ctx.Corners(job);
    device uint *opposites = ctx.Opposites(job);
    for (uint i = first; i < last; ++i) {
        const uint h = items[i];
        const uint key = ConnBucketKey(corners, h);
        uint opposite = ConnNullHalfedge, sharing = 0u;
        for (uint j = first; j < last; ++j) {
            const uint other = ConnBucketKey(corners, items[j]);
            if (other == (key ^ ConnReverseBit) && opposite == ConnNullHalfedge) opposite = items[j];
            sharing += (other & ~ConnReverseBit) == (key & ~ConnReverseBit) ? 1u : 0u;
        }
        // Three halfedges on one edge cannot be represented by one halfedge/opposite pair, so the CPU rebuilds this mesh.
        if (sharing > 2u) atomic_fetch_or_explicit(&ctx.AtomicScratch()[job.StateOffset + 1u], 1u, memory_order_relaxed);
        if (opposite < h) {
            opposites[h] = opposite;
            opposites[opposite] = h;
        }
    }
}

kernel void MeshConnectivityBits(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint word = tile.y * ScanTileSize + lane;
    if (word > job.WordCount) return;
    if (word == job.WordCount) {
        ctx.Scratch()[job.PopcountOffset + word] = 0u;
        return;
    }
    device const uint *opposites = ctx.Opposites(job);
    uint bits = 0u;
    for (uint b = 0u; b < 32u; ++b) {
        const uint h = word * 32u + b;
        if (h >= job.HalfedgeCount) break;
        // Assign the edge index to the lowest halfedge in each opposite pair.
        const uint opposite = opposites[h];
        if (opposite == ConnNullHalfedge || opposite > h) bits |= 1u << b;
    }
    ctx.EdgeFirstBits(job)[word] = bits;
    ctx.Scratch()[job.PopcountOffset + word] = popcount(bits);
}

kernel void MeshConnectivityWordBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    ScanBlockSum(
        ctx.Scratch() + job.PopcountOffset, job.WordCount + 1u, tile.y, ctx.Scratch() + job.WordBlockOffset,
        lane, simd_lane, simd_group, sums
    );
}

kernel void MeshConnectivityWordBlockPrefix(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const MeshConnectivityJob job = ctx.Jobs()[group_id];
    ScanBlockPrefix(ctx.Scratch() + job.WordBlockOffset, job.WordBlockCount, lane, simd_lane, simd_group, sums);
}

kernel void MeshConnectivityRanks(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    uint local[ScanPerThread];
    uint start = ScanBlockStart(
        ctx.Scratch() + job.PopcountOffset, job.WordCount + 1u, tile.y, ctx.Scratch() + job.WordBlockOffset,
        lane, simd_lane, simd_group, sums, local
    );
    device uint *ranks = ctx.EdgeFirstRanks(job);
    const uint base = tile.y * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint word = base + k;
        if (word > job.WordCount) break;
        // Store the final prefix total as the edge count.
        if (word < job.WordCount) ranks[word] = start;
        else ctx.Scratch()[job.StateOffset] = start;
        start += local[k];
    }
}

kernel void MeshConnectivitySamples(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshConnectivityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint sample = tile.y * ScanTileSize + lane;
    const uint edge_count = ctx.Scratch()[job.StateOffset];
    if (sample * 32u >= edge_count) return;
    // Each sample records the word containing edge 32 * sample to bound rank lookup.
    device const uint *ranks = ctx.EdgeFirstRanks(job);
    const uint edge = sample * 32u;
    uint low = 0u, high = job.WordCount - 1u;
    while (low < high) {
        const uint mid = (low + high + 1u) / 2u;
        if (ranks[mid] <= edge) low = mid;
        else high = mid - 1u;
    }
    ctx.EdgeSamples(job)[sample] = low;
}

#endif
