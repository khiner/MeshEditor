#ifndef MESHCONNECTIVITY_MSL
#define MESHCONNECTIVITY_MSL

// Builds vertex-outgoing halfedges, opposites, each halfedge's edge, and each edge's first halfedge for one mesh.
// Face halfedges hash by endpoint pair into an open-addressed table whose slot holds the lowest halfedge of each undirected edge.
// A wire corner has no face owner and names its other endpoint through its explicit pair.
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

    device uint *Storage() const { return BindlessBufferMutable(uint, B.Buffer, Pc.StorageSlot); }
    device const MeshConnectivityJob *Jobs() const { return reinterpret_cast<device const MeshConnectivityJob *>(Storage() + Pc.JobsOffset); }
    device const uint2 *Tiles() const { return reinterpret_cast<device const uint2 *>(Storage() + Pc.TileMapOffset); }
    device uint *Scratch() const { return Storage() + Pc.ScratchOffset; }
    device atomic_uint *AtomicScratch() const { return reinterpret_cast<device atomic_uint *>(Scratch()); }
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
        if (f == InvalidOffset) return Opposites(job)[h];
        device const uint *range = Ranges(job) + 2u * f;
        return h == range[0] ? range[1] - 1u : h - 1u;
    }

    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }
};

inline uint2 ConnEndpoints(ConnContext ctx, MeshConnectivityJob job, device const uint *corners, uint h) { return uint2(corners[ctx.Prev(job, h)], corners[h]); }

// Faces share endpoint keys; explicit wire pairs remain distinct, including coincident lines.
inline uint3 ConnEdgeKey(uint2 ends, uint wire_pair = 0u) { return uint3(min(ends.x, ends.y), max(ends.x, ends.y), wire_pair); }
inline uint3 ConnKey(ConnContext ctx, MeshConnectivityJob job, device const uint *corners, uint h) {
    const uint pair = ctx.Owners(job)[h] == InvalidOffset ? min(h, ctx.Opposites(job)[h]) + 1u : 0u;
    return ConnEdgeKey(ConnEndpoints(ctx, job, corners, h), pair);
}

// A halfedge runs in reverse when it leaves the higher endpoint.
inline bool ConnReverse(uint2 ends) { return ends.x > ends.y; }

inline uint ConnEdgeHash(uint3 key) {
    uint hash = key.x * 0x9E3779B1u;
    hash ^= key.y * 0x85EBCA77u;
    hash ^= key.z * 0x27D4EB2Du;
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
    const uint2 ends = ConnEndpoints(ctx, job, corners, h);
    const uint3 key = ConnKey(ctx, job, corners, h);
    // A closure may include corners belonging to vertices outside the writable
    // vertex domain. Their incidence lists and outgoing handles remain intact.
    if (ctx.Vertices(job).Index(ends.x) != InvalidOffset) atomic_fetch_min_explicit(&ctx.AtomicOutgoing(job)[ends.x], h, memory_order_relaxed);
    device atomic_uint *table = ctx.AtomicScratch() + job.TableOffset;
    // Equal keys share a probe sequence, and atomic min selects their lowest halfedge.
    uint slot = ConnEdgeHash(key) & job.TableMask;
    for (;;) {
        uint occupant = ConnEmptySlot;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &occupant, h, memory_order_relaxed, memory_order_relaxed)) break;
        if (occupant == ConnEmptySlot) continue;
        if (all(ConnKey(ctx, job, corners, occupant) == key)) {
            atomic_fetch_min_explicit(&table[slot], h, memory_order_relaxed);
            break;
        }
        slot = (slot + 1u) & job.TableMask;
    }
    ctx.Rep(job)[i] = slot;
}

// Match the old affected edges against the new endpoint-pair table. Source
// bindings may name page clones while destination corners/owners are live.
// Face edges match endpoint pairs. A wire keeps its explicit pair identity,
// including coincident lines and zero-length input edges.
kernel void MeshConnectivityMatchEdges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshConnectivityJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    uint match=InvalidOffset;
    if (i < job.SourceEdgeCount) {
        const uint e = ctx.SourceEdges(job).Handle(i);
        const uint h = BindlessBuffer(uint,bindless.Buffer,job.SourceConnectivity.Edges.Slot)[e];
        const uint f = BindlessBuffer(uint,bindless.Buffer,job.SourceConnectivity.HalfedgeFaces.Slot)[h];
        uint previous;
        if (f == InvalidOffset) previous = BindlessBuffer(uint, bindless.Buffer, job.SourceConnectivity.Opposites.Slot)[h];
        else {
            const auto ranges = BindlessBuffer(packed_uint2,bindless.Buffer,job.SourceConnectivity.FaceRanges.Slot);
            const uint2 loop = uint2(ranges[f]);
            previous = h == loop.x ? loop.y - 1u : h - 1u;
        }
        const auto corners = BindlessBuffer(uint,bindless.IndexBuffer,job.SourceCornerSlot);
        const uint converted=job.ConvertedEdges.Storage.Slot!=InvalidSlot ? WorkRank(bindless,job.ConvertedEdges,e) : InvalidOffset;
        uint pair=f==InvalidOffset ? min(h,previous)+1u : 0u;
        if (converted!=InvalidOffset) pair=job.ConvertedWirePairs.Slot!=InvalidSlot ? ctx.Words(job.ConvertedWirePairs)[job.ConvertedWirePairs.Offset+converted] : 0u;
        const uint3 key = ConnEdgeKey(uint2(corners[previous], corners[h]),pair);
        uint slot = ConnEdgeHash(key) & job.TableMask;
        for (uint probe = 0u; pair!=InvalidOffset && probe <= job.TableMask; ++probe, slot = (slot + 1u) & job.TableMask) {
            const uint representative = ctx.Scratch()[job.TableOffset + slot];
            if (representative == ConnEmptySlot) break;
            if (all(ConnKey(ctx, job, ctx.Corners(job), representative) == key)) {
                match=ctx.Halfedges(job).Index(representative);
                atomic_fetch_min_explicit(ctx.AtomicScratch()+job.RetainedEdgesOffset+match,e,memory_order_relaxed);
                break;
            }
        }
        // Reuse the eventual retired-edge list until its compaction pass.
        ctx.Scratch()[job.RetiredEdgesOffset+i]=match;
    }
}

// Several source edges can become one surface edge. Retain one deterministic
// handle and retire the other claimants after all matches have been published.
kernel void MeshConnectivityClassifyEdges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const ConnContext ctx{bindless,pc};
    const uint2 tile=ctx.Tile(group_id);
    const MeshConnectivityJob job=ctx.Jobs()[tile.x];
    const uint i=tile.y*ScanTileSize+lane;
    bool retired=false;
    if (i<job.SourceEdgeCount) {
        const uint match=ctx.Scratch()[job.RetiredEdgesOffset+i];
        retired=match==InvalidOffset || ctx.RetainedEdge(job,match)!=ctx.SourceEdges(job).Handle(i);
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
    const bool wire = ctx.Owners(job)[h] == InvalidOffset;
    const bool reverse = wire ? h > ctx.Opposites(job)[h] : ConnReverse(ConnEndpoints(ctx, job, corners, h));
    // The partner is the lowest halfedge running against the representative.
    if (reverse != (wire ? representative > ctx.Opposites(job)[representative] : ConnReverse(ConnEndpoints(ctx, job, corners, representative)))) {
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
