#ifndef MESHCONNECTIVITY_MSL
#define MESHCONNECTIVITY_MSL

// Builds vertex-outgoing halfedges, opposites, each halfedge's edge, and each edge's first halfedge for one mesh.
// Face halfedges hash by endpoint pair into an open-addressed table whose slot holds the lowest halfedge of each undirected edge.
// A mesh without faces holds line corners, which pair in consecutive work order, each pair one edge.
// Each edge links its lowest forward halfedge to its lowest reverse halfedge and leaves every other incidence unlinked.
// Edge ranks follow compact work order.
// An explicit map preserves caller-owned edge handles.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
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
    device const uint *Corners(MeshConnectivityJob job) const { return BindlessBuffer(uint, B.IndexBuffer, job.Corners.Slot); }
    device uint *Words(SlotOffset at) const { return BindlessBufferMutable(uint, B.Buffer, at.Slot); }
    device uint *Outgoing(MeshConnectivityJob job) const { return Words(job.Connectivity.Outgoing); }
    device atomic_uint *AtomicOutgoing(MeshConnectivityJob job) const {
        return BindlessBufferMutable(atomic_uint, B.Buffer, job.Connectivity.Outgoing.Slot);
    }
    device uint *Opposites(MeshConnectivityJob job) const { return Words(job.Connectivity.Opposites); }
    device uint *HalfedgeToEdge(MeshConnectivityJob job) const { return Words(job.Connectivity.HalfedgeEdges); }
    device uint *Owners(MeshConnectivityJob job) const { return Words(job.Connectivity.HalfedgeFaces); }
    device uint *Ranges(MeshConnectivityJob job) const { return BindlessBufferMutable(uint, B.Buffer, job.Connectivity.FaceRanges.Slot); }
    device uint *Edges(MeshConnectivityJob job) const { return Words(job.Connectivity.Edges); }
    device uint *EdgeFirstBits(MeshConnectivityJob job) const { return Scratch() + job.BitsOffset; }
    device uint *EdgeFirstRanks(MeshConnectivityJob job) const { return Scratch() + job.RanksOffset; }
    device uint *Rep(MeshConnectivityJob job) const { return Scratch() + job.RepOffset; }
    device uint *Partner(MeshConnectivityJob job) const { return Scratch() + job.PartnerOffset; }
    device atomic_uint *AtomicPartner(MeshConnectivityJob job) const { return AtomicScratch() + job.PartnerOffset; }
    ElementWorkDomain Vertices(MeshConnectivityJob job) const { return {B, job.Vertices, job.Vertices.Storage.Slot == InvalidSlot ? job.Connectivity.Outgoing.Offset : 0u}; }
    ElementWorkDomain Halfedges(MeshConnectivityJob job) const { return {B, job.Halfedges, job.Halfedges.Storage.Slot == InvalidSlot ? job.Corners.Offset : 0u}; }
    ElementWorkDomain Faces(MeshConnectivityJob job) const { return {B, job.Faces, job.Faces.Storage.Slot == InvalidSlot ? job.Connectivity.FaceRanges.Offset : 0u}; }
    ElementWorkDomain SourceEdges(MeshConnectivityJob job) const { return {B, job.SourceEdges, job.SourceEdges.Storage.Slot == InvalidSlot ? job.SourceConnectivity.Edges.Offset : 0u}; }
    uint RetainedEdge(MeshConnectivityJob job, uint representative_index) const {
        return job.RetainedEdgesOffset == InvalidOffset ? InvalidOffset : Scratch()[job.RetainedEdgesOffset + representative_index];
    }
    uint EdgeHandle(MeshConnectivityJob job, uint index) const {
        const ElementHandleRange edges = job.EdgeHandles;
        return edges.Handles.Slot == InvalidSlot ? edges.First + index : BindlessBuffer(uint, B.Buffer, edges.Handles.Slot)[edges.Handles.Offset + index];
    }
    uint Prev(MeshConnectivityJob job, uint h) const {
        const uint f = Owners(job)[h];
        device const uint *range = Ranges(job) + 2u * f;
        return h == range[0] ? range[1] - 1u : h - 1u;
    }

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

// Face loops own their canonical corners independently of face allocation order.
// Only packed triangle import needs implicit ranges.
// Emitters supply explicit ranges.
kernel void MeshConnectivityFaces(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.FaceCount) return;
    const uint f = ctx.Faces(job).Handle(i);
    device uint *range = ctx.Ranges(job) + 2u * f;
    if (job.FaceStarts == 0u) {
        range[0] = job.Corners.Offset + 3u * i;
        range[1] = range[0] + 3u;
    }
    for (uint h = range[0]; h < range[1]; ++h)
        if (ctx.Halfedges(job).Index(h) != InvalidOffset) ctx.Owners(job)[h] = f;
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
    if (i < job.VertexCount) {
        const uint v = ctx.Vertices(job).Handle(i);
        ctx.Outgoing(job)[v] = ConnNullHalfedge;
    }
    if (i < job.HalfedgeCount) {
        ctx.Partner(job)[i] = ConnNullHalfedge;
        if (job.FaceCount == 0u) ctx.Owners(job)[ctx.Halfedges(job).Handle(i)] = InvalidOffset;
        if (job.RetainedEdgesOffset != InvalidOffset) ctx.Scratch()[job.RetainedEdgesOffset + i] = InvalidOffset;
    }
    if (i == 0u) {
        ctx.Scratch()[job.PopcountOffset + job.WordCount] = 0u;
        ctx.Scratch()[job.PopcountOffset + job.ScanWordCount - 1u] = 0u;
    }
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
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.HalfedgeCount) return;
    const uint h = ctx.Halfedges(job).Handle(i);
    device const uint *corners = ctx.Corners(job);
    const uint2 ends = job.FaceCount == 0u ? uint2(corners[ctx.Halfedges(job).Handle(i ^ 1u)], corners[h]) : ConnEndpoints(ctx, job, corners, h);
    const uint2 key = ConnEdgeKey(ends);
    // A closure may include corners belonging to vertices outside the writable
    // vertex domain. Their incidence lists and outgoing handles remain intact.
    if (ctx.Vertices(job).Index(ends.x) != InvalidOffset) atomic_fetch_min_explicit(&ctx.AtomicOutgoing(job)[ends.x], h, memory_order_relaxed);
    if (job.FaceCount == 0u) {
        const uint pair = i / 2u;
        if ((i & 1u) == 0u) ctx.Scratch()[job.TableOffset + pair] = h;
        ctx.Rep(job)[i] = pair;
        return;
    }
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
    ctx.Rep(job)[i] = slot;
}

// Match the old affected edges against the new endpoint-pair table. Source
// bindings may name page clones while destination corners/owners are live.
// Canonical source connectivity has exactly one edge per unordered endpoint pair.
kernel void MeshConnectivityMatchEdges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    bool retired = false;
    if (i < job.SourceEdgeCount && job.FaceCount == 0u) {
        // A line edge's first corner leads its pair, so the edge stays exactly when that corner stays in the work.
        const uint e = ctx.SourceEdges(job).Handle(i);
        const uint rank = ctx.Halfedges(job).Index(BindlessBuffer(uint,bindless.Buffer,job.SourceConnectivity.Edges.Slot)[e]);
        if (rank != InvalidOffset) ctx.Scratch()[job.RetainedEdgesOffset + rank] = e;
        retired = rank == InvalidOffset;
    } else if (i < job.SourceEdgeCount) {
        const uint e = ctx.SourceEdges(job).Handle(i);
        const uint h = BindlessBuffer(uint,bindless.Buffer,job.SourceConnectivity.Edges.Slot)[e];
        const uint f = BindlessBuffer(uint,bindless.Buffer,job.SourceConnectivity.HalfedgeFaces.Slot)[h];
        const auto ranges = BindlessBuffer(packed_uint2,bindless.Buffer,job.SourceConnectivity.FaceRanges.Slot);
        const uint2 loop = uint2(ranges[f]);
        const auto corners = BindlessBuffer(uint,bindless.IndexBuffer,job.SourceCornerSlot);
        const uint2 key = ConnEdgeKey(uint2(corners[h == loop.x ? loop.y - 1u : h - 1u], corners[h]));
        uint slot = ConnEdgeHash(key) & job.TableMask;
        retired = true;
        for (uint probe = 0u; probe <= job.TableMask; ++probe, slot = (slot + 1u) & job.TableMask) {
            const uint representative = ctx.Scratch()[job.TableOffset + slot];
            if (representative == ConnEmptySlot) break;
            if (all(ConnEdgeKey(ConnEndpoints(ctx, job, ctx.Corners(job), representative)) == key)) {
                ctx.Scratch()[job.RetainedEdgesOffset + ctx.Halfedges(job).Index(representative)] = e;
                retired = false;
                break;
            }
        }
    }
    const uint bits = uint((simd_vote::vote_t)simd_ballot(retired));
    const uint word = tile.y * ScanSimdGroups + simd_group;
    if (simd_lane == 0u && word < job.SourceWordCount) {
        ctx.Scratch()[job.BitsOffset + job.WordCount + word] = bits;
        ctx.Scratch()[job.PopcountOffset + job.WordCount + 1u + word] = popcount(bits);
    }
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
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.HalfedgeCount) return;
    const uint h = ctx.Halfedges(job).Handle(i);
    device uint *rep = ctx.Rep(job);
    const uint representative = ctx.Scratch()[job.TableOffset + rep[i]];
    rep[i] = representative;
    device const uint *corners = ctx.Corners(job);
    if (job.FaceCount == 0u) {
        if (i & 1u) ctx.Partner(job)[ctx.Halfedges(job).Index(representative)] = h;
        return;
    }
    const bool reverse = ConnReverse(ConnEndpoints(ctx, job, corners, h));
    // The partner is the lowest halfedge running against the representative.
    if (reverse != ConnReverse(ConnEndpoints(ctx, job, corners, representative))) {
        atomic_fetch_min_explicit(&ctx.AtomicPartner(job)[ctx.Halfedges(job).Index(representative)], h, memory_order_relaxed);
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
    const uint i = tile.y * ScanTileSize + lane;
    device uint *scratch = ctx.Scratch();
    bool first = false;
    if (i < job.HalfedgeCount) {
        const uint h = ctx.Halfedges(job).Handle(i);
        const uint representative = ctx.Rep(job)[i];
        const uint partner = ctx.Partner(job)[ctx.Halfedges(job).Index(representative)];
        const uint opposite = h == representative ? partner : (h == partner ? representative : ConnNullHalfedge);
        ctx.Opposites(job)[h] = opposite;
        first = representative == h && ctx.RetainedEdge(job, i) == InvalidOffset;
    }
    // Every lane votes, so the ballot is the mark word of this simdgroup's 32 consecutive halfedges.
    const uint bits = uint((simd_vote::vote_t)simd_ballot(first));
    const uint word = tile.y * ScanSimdGroups + simd_group;
    if (simd_lane == 0u && word < job.WordCount) {
        ctx.EdgeFirstBits(job)[word] = bits;
        scratch[job.PopcountOffset + word] = popcount(bits);
    }
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
        ctx.Scratch() + job.PopcountOffset, job.ScanWordCount, tile.y, ctx.Scratch() + job.WordBlockOffset,
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
    ScanBlockOffsets(
        ctx.Scratch() + job.PopcountOffset, job.ScanWordCount, tile.y, ctx.Scratch() + job.WordBlockOffset, ctx.EdgeFirstRanks(job),
        lane, simd_lane, simd_group, sums
    );
}

// Both terminal prefixes are available after the rank pass, including when
// their words were processed by different threadgroups.
kernel void MeshConnectivityCounts(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    const ConnContext ctx{bindless, pc};
    const MeshConnectivityJob job = ctx.Jobs()[group_id];
    const uint added = ctx.EdgeFirstRanks(job)[job.WordCount];
    ctx.Scratch()[job.StateOffset] = added;
    ctx.Scratch()[job.StateOffset + 1u] = ctx.EdgeFirstRanks(job)[job.ScanWordCount - 1u] - added;
}

kernel void MeshConnectivityRetiredEdges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.SourceEdgeCount) return;
    const uint word = i >> 5u, bit = i & 31u;
    const uint bits = ctx.Scratch()[job.BitsOffset + job.WordCount + word];
    if (!(bits & (1u << bit))) return;
    const uint rank = ctx.EdgeFirstRanks(job)[job.WordCount + 1u + word] - ctx.Scratch()[job.StateOffset] +
        popcount(bits & ((1u << bit) - 1u));
    ctx.Scratch()[job.RetiredEdgesOffset + rank] = ctx.SourceEdges(job).Handle(i);
    if (job.RetiredEdgeWork.Storage.Slot != InvalidSlot) MarkWork(bindless,job.RetiredEdgeWork,ctx.SourceEdges(job).Handle(i));
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
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= job.HalfedgeCount) return;
    const uint h = ctx.Halfedges(job).Handle(i);
    const uint representative = ctx.Rep(job)[i];
    const uint rank = ctx.Halfedges(job).Index(representative);
    const uint word = rank >> 5u;
    const uint edge = ctx.EdgeFirstRanks(job)[word] + popcount(ctx.EdgeFirstBits(job)[word] & ((1u << (rank & 31u)) - 1u));
    const uint retained = ctx.RetainedEdge(job, rank);
    const uint handle = retained == InvalidOffset ? ctx.EdgeHandle(job, edge) : retained;
    ctx.HalfedgeToEdge(job)[h] = handle;
    if (representative == h) ctx.Edges(job)[handle] = h;
}


#endif
