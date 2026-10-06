#include "Bindless.metal"
#include "ElementWorkShared.metal"
#include "RadixSort.metal"
#include "gpu/VertexFanBuildJob.h"

// A fresh job keeps one count word per vertex and one rank key per corner.
// Another job keeps each vertex's handle, count and former root, and each corner's handle and rank.
struct FanBuildContext {
    device const BindlessSet &B;
    constant VertexFanBuildPushConstants &Pc;
    device uint *Storage() const { return BindlessBufferMutable(uint,B.Buffer,Pc.StorageSlot); }
    device uint *Scratch() const { return Storage()+Pc.ScratchOffset; }
    uint2 Tile(uint group) const { return reinterpret_cast<device const uint2 *>(Storage()+Pc.TileMapOffset)[Pc.FirstTile+group]; }
    VertexFanBuildJob Job(uint index) const { return reinterpret_cast<device const VertexFanBuildJob *>(Storage()+Pc.JobsOffset)[index]; }
    ElementWorkDomain Vertices(VertexFanBuildJob job) const { return {B,job.Vertices,job.Vertices.Storage.Slot == InvalidSlot ? job.Roots.Offset : 0u}; }
    ElementWorkDomain Halfedges(VertexFanBuildJob job) const { return {B,job.Halfedges,job.Halfedges.Storage.Slot == InvalidSlot ? job.Corners.Offset : 0u}; }
    uint CountWord(VertexFanBuildJob job, uint i) const { return job.Metadata+(job.Fresh ? i : 4u*i+1u); }
    RadixSortView Sort(VertexFanBuildJob job) const {
        device uint *s = Scratch();
        return {s+job.Keys,s+job.Order,s+job.Temporary,s+job.Histogram,s+job.Totals,
            job.HalfedgeCount,(job.HalfedgeCount+255u)/256u,job.Fresh ? 1u : 2u,
            job.Fresh ? 0u : 1u,Pc.PassParameter*4u,bool(Pc.PassParameter&1u)};
    }
};

kernel void VertexFanInit(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile = ctx.Tile(group); const auto job = ctx.Job(tile.x);
    const uint i = tile.y*256u+lane;
    if (i < job.VertexCount) {
        device uint *metadata = ctx.Scratch()+job.Metadata;
        if (job.Fresh) metadata[i] = 0u;
        else {
            const uint v = ctx.Vertices(job).Handle(i);
            const uint2 root = uint2(BindlessBuffer(packed_uint2,b.Buffer,job.Roots.Slot)[v]);
            metadata[4u*i] = v;
            metadata[4u*i+1u] = 0u;
            metadata[4u*i+2u] = root.x;
            metadata[4u*i+3u] = root.y;
        }
    }
    if (i < job.HalfedgeCount) ctx.Scratch()[job.Order+i] = i;
}
kernel void VertexFanKeys(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile = ctx.Tile(group); const auto job = ctx.Job(tile.x);
    const uint i = tile.y*256u+lane;
    if (i >= job.HalfedgeCount) return;
    const uint h = ctx.Halfedges(job).Handle(i);
    const uint v = BindlessBuffer(uint,b.IndexBuffer,job.Corners.Slot)[h];
    const uint at = ctx.Vertices(job).Index(v);
    if (job.Fresh) ctx.Scratch()[job.Keys+i]=at;
    else { ctx.Scratch()[job.Keys+2u*i]=h; ctx.Scratch()[job.Keys+2u*i+1u]=at; }
    if (at != InvalidOffset) atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(ctx.Scratch()+ctx.CountWord(job,at)),1u,memory_order_relaxed);
}
// Each vertex tile's corner count, then the exclusive prefix of the tiles, whose last word is the job's total.
kernel void VertexFanTileScan(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile=ctx.Tile(group); const auto job=ctx.Job(tile.x);
    const uint i=tile.y*256u+lane;
    if (tile.y >= (job.VertexCount+255u)/256u) return;
    const uint count=i<job.VertexCount ? ctx.Scratch()[ctx.CountWord(job,i)] : 0u;
    threadgroup uint sums[9];
    ThreadgroupExclusiveScan(count,lane,sl,sg,sums);
    if (lane==0u) ctx.Scratch()[job.TileData+tile.y]=sums[8];
}
kernel void VertexFanTilePrefix(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const FanBuildContext ctx{b,pc}; const auto job=ctx.Job(group);
    device uint *tiles=ctx.Scratch()+job.TileData;
    const uint count=(job.VertexCount+255u)/256u;
    if (lane==0u) tiles[count]=0u;
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint sums[9];
    ScanBlockPrefix(tiles,count+1u,lane,sl,sg,sums);
}
kernel void VertexFanHistogram(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile = ctx.Tile(group);
    const auto job=ctx.Job(tile.x);
    if (pc.PassParameter>=job.VertexKeyPasses) return;
    threadgroup atomic_uint counts[16];
    RadixHistogram(ctx.Sort(job),lane,tile.y,counts);
}
kernel void VertexFanPrefix(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile = ctx.Tile(group);
    const auto job=ctx.Job(tile.x);
    if (pc.PassParameter>=job.VertexKeyPasses) return;
    threadgroup uint sums[9];
    RadixPrefix(ctx.Sort(job),lane,tile.y,sl,sg,sums);
}
kernel void VertexFanScatter(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile = ctx.Tile(group);
    const auto job=ctx.Job(tile.x);
    if (pc.PassParameter>=job.VertexKeyPasses) return;
    threadgroup uint groups[128];
    RadixScatter(ctx.Sort(job),lane,tile.y,sl,sg,groups);
}
// Corners sorted by vertex rank fill the run in order, so each vertex's root starts at its rank's prefix.
kernel void VertexFanEmit(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexFanBuildPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const FanBuildContext ctx{b,pc}; const uint2 tile=ctx.Tile(group); const auto job=ctx.Job(tile.x);
    const uint i=tile.y*256u+lane;
    device uint *scratch=ctx.Scratch();
    if (i<job.HalfedgeCount) {
        const uint source=scratch[(job.VertexKeyPasses & 1u ? job.Temporary : job.Order)+i];
        const uint v=job.Fresh ? scratch[job.Keys+source] : scratch[job.Keys+2u*source+1u];
        if (v!=InvalidOffset) {
            const uint h=job.Fresh ? job.Corners.Offset+source : scratch[job.Keys+2u*source];
            const uint face=job.FaceCount ? BindlessBuffer(uint,b.Buffer,job.FaceOwnersSlot)[h] : InvalidOffset;
            BindlessBufferMutable(packed_uint2,b.Buffer,job.ItemsSlot)[job.FirstItem+i]=packed_uint2(h,face);
        }
    }
    if (tile.y<(job.VertexCount+255u)/256u) {
        const uint count=i<job.VertexCount ? scratch[ctx.CountWord(job,i)] : 0u;
        threadgroup uint sums[9];
        const uint local=ThreadgroupExclusiveScan(count,lane,sl,sg,sums);
        if (i<job.VertexCount) {
            const uint offset=count ? job.FirstItem+scratch[job.TileData+tile.y]+local : InvalidOffset;
            const uint root=job.Fresh ? job.Roots.Offset+i : scratch[job.Metadata+4u*i];
            BindlessBufferMutable(packed_uint2,b.Buffer,job.Roots.Slot)[root]=packed_uint2(offset,count);
        }
    }
}
