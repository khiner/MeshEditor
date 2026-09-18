#ifndef MESHCONNECTIVITY_MSL
#define MESHCONNECTIVITY_MSL

// Builds vertex-outgoing halfedges, opposites, each halfedge's edge, and each edge's first halfedge for one face mesh.
// Halfedges hash by endpoint pair into an open-addressed table whose slot holds the lowest halfedge of each undirected edge.
// Each edge links its lowest forward halfedge to its lowest reverse halfedge and leaves every other incidence unlinked.
// Edges number by ascending representative halfedge, the lowest halfedge sharing the edge.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "gpu/MeshConnectivityJob.h"
#include "gpu/TiledJobPushConstants.h"

constant uint ConnNullHalfedge = InvalidOffset;
constant uint ConnEmptySlot = InvalidOffset;

struct ConnContext {
    device const BindlessSet &B;
    constant TiledJobPushConstants &Pc;

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
    device uint *HalfedgeToEdge(MeshConnectivityJob job) const { return Run(job) + job.VertexCount + job.HalfedgeCount; }
    device uint *Edges(MeshConnectivityJob job) const { return Run(job) + job.VertexCount + 2u * job.HalfedgeCount + (job.FaceStarts != 0u ? job.FaceCount : 0u); }
    device uint *EdgeFirstBits(MeshConnectivityJob job) const { return Scratch() + job.BitsOffset; }
    device uint *EdgeFirstRanks(MeshConnectivityJob job) const { return Scratch() + job.RanksOffset; }
    device uint *Rep(MeshConnectivityJob job) const { return Scratch() + job.RepOffset; }
    device uint *Partner(MeshConnectivityJob job) const { return Scratch() + job.PartnerOffset; }
    device atomic_uint *AtomicPartner(MeshConnectivityJob job) const { return AtomicScratch() + job.PartnerOffset; }
    // The halfedge before `h` in its face loop: arithmetic for a triangle mesh, staged otherwise.
    uint Prev(MeshConnectivityJob job, uint h) const { return job.PrevOffset == InvalidOffset ? ConnectivityPrevious(h) : Scratch()[job.PrevOffset + h]; }

    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

inline uint2 ConnEndpoints(ConnContext ctx, MeshConnectivityJob job, device const uint *corners, uint h) { return uint2(corners[ctx.Prev(job, h)], corners[h]); }

// The undirected edge of a halfedge, as its endpoints in ascending order.
inline uint2 ConnEdgeKey(uint2 ends) { return uint2(min(ends.x, ends.y), max(ends.x, ends.y)); }

// A halfedge runs in reverse when it leaves the higher endpoint.
inline bool ConnReverse(uint2 ends) { return ends.x > ends.y; }

inline uint ConnEdgeHash(uint2 key) {
    uint hash = key.x * 0x9E3779B1u;
    hash ^= key.y * 0x85EBCA77u;
    hash ^= hash >> 15u;
    hash *= 0xC2B2AE3Du;
    hash ^= hash >> 13u;
    return hash;
}

// Stages each halfedge's predecessor for a mesh whose faces are not all triangles.
kernel void MeshConnectivityPrev(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (job.PrevOffset == InvalidOffset || h >= job.HalfedgeCount) return;
    const ConnectivityView conn{ctx.Run(job), job.VertexCount, job.HalfedgeCount, job.FaceCount, job.FaceStarts != 0u};
    ctx.Scratch()[job.PrevOffset + h] = conn.Previous(h);
}

// Empties the edge table and nulls every outgoing halfedge and partner.
kernel void MeshConnectivityInit(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i <= job.TableMask) ctx.Scratch()[job.TableOffset + i] = ConnEmptySlot;
    if (i < job.VertexCount) ctx.Outgoing(job)[i] = ConnNullHalfedge;
    if (i < job.HalfedgeCount) ctx.Partner(job)[i] = ConnNullHalfedge;
}

// Claims or joins each halfedge's edge slot and records the slot per halfedge.
kernel void MeshConnectivityInsert(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    device const uint *corners = ctx.Corners(job);
    const uint2 ends = ConnEndpoints(ctx, job, corners, h);
    const uint2 key = ConnEdgeKey(ends);
    // Select the lowest outgoing halfedge to match the CPU store.
    atomic_fetch_min_explicit(&ctx.AtomicOutgoing(job)[ends.x], h, memory_order_relaxed);
    device atomic_uint *table = ctx.AtomicScratch() + job.TableOffset;
    // Equal keys share a probe sequence, and atomic min selects their lowest halfedge.
    uint slot = ConnEdgeHash(key) & job.TableMask;
    for (;;) {
        uint occupant = ConnEmptySlot;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &occupant, h, memory_order_relaxed, memory_order_relaxed)) break;
        if (occupant == ConnEmptySlot) continue;
        if (all(ConnEdgeKey(ConnEndpoints(ctx, job, corners, occupant)) == key)) {
            atomic_fetch_min_explicit(&table[slot], h, memory_order_relaxed);
            break;
        }
        slot = (slot + 1u) & job.TableMask;
    }
    ctx.Rep(job)[h] = slot;
}

// Replaces each halfedge's slot with its edge's representative and offers itself as the representative's partner.
kernel void MeshConnectivityResolve(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    device uint *rep = ctx.Rep(job);
    const uint representative = ctx.Scratch()[job.TableOffset + rep[h]];
    rep[h] = representative;
    device const uint *corners = ctx.Corners(job);
    const bool reverse = ConnReverse(ConnEndpoints(ctx, job, corners, h));
    // The partner is the lowest halfedge running against the representative.
    if (reverse != ConnReverse(ConnEndpoints(ctx, job, corners, representative))) {
        atomic_fetch_min_explicit(&ctx.AtomicPartner(job)[representative], h, memory_order_relaxed);
    }
}

// Links each representative with its partner and marks the representatives per halfedge word.
kernel void MeshConnectivityLink(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    device uint *scratch = ctx.Scratch();
    bool first = false;
    if (h < job.HalfedgeCount) {
        const uint representative = ctx.Rep(job)[h];
        const uint partner = ctx.Partner(job)[representative];
        ctx.Opposites(job)[h] = h == representative ? partner : (h == partner ? representative : ConnNullHalfedge);
        // The edge index goes to the lowest halfedge of each edge.
        first = representative == h;
    }
    // Every lane votes, so the ballot is the mark word of this simdgroup's 32 consecutive halfedges.
    const uint bits = uint((simd_vote::vote_t)simd_ballot(first));
    const uint word = tile.y * ScanSimdGroups + simd_group;
    if (simd_lane == 0u && word < job.WordCount) {
        ctx.EdgeFirstBits(job)[word] = bits;
        scratch[job.PopcountOffset + word] = popcount(bits);
    }
    // The unmarked terminator receives the edge count from the exclusive scan.
    if (h == 0u) scratch[job.PopcountOffset + job.WordCount] = 0u;
}

kernel void MeshConnectivityWordBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
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
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
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
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
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

// Numbers each halfedge's edge by its representative's rank and records each edge's first halfedge.
kernel void MeshConnectivityEdgeTables(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.HalfedgeCount) return;
    const uint representative = ctx.Rep(job)[h];
    const uint word = representative >> 5u;
    const uint edge = ctx.EdgeFirstRanks(job)[word] + popcount(ctx.EdgeFirstBits(job)[word] & ((1u << (representative & 31u)) - 1u));
    ctx.HalfedgeToEdge(job)[h] = edge;
    if (representative == h) ctx.Edges(job)[edge] = h;
}

#endif
