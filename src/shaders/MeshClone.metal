#include "MeshletIndexShared.metal"
#include "gpu/MeshCloneJob.h"
// Copies each run sixteen bytes per thread, a whole vector when the run is aligned to it.
kernel void CopyByteRuns(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const ByteCopy *jobs [[buffer(CloneBufferIndex_Jobs)]], device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const ByteCopy job=jobs[tile.x];
    const ulong first=ulong(tile.y)*CloneCopyTileBytes+ulong(lane)*CloneCopyThreadBytes;
    if (first>=job.Bytes) return;
    if (((job.Source|job.Destination|job.Bytes)&ulong(CloneCopyThreadBytes-1u))==0ul) {
        *(device uint4 *)(bytes+job.Destination+first)=*(device const uint4 *)(bytes+job.Source+first);
        return;
    }
    for (ulong i=first, end=min(first+CloneCopyThreadBytes,job.Bytes); i<end; ++i) bytes[job.Destination+i]=bytes[job.Source+i];
}
kernel void RebaseByBlock(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const BlockRebase *jobs [[buffer(CloneBufferIndex_Jobs)]],
    device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]], device const uint2 *maps [[buffer(CloneBufferIndex_Maps)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const BlockRebase job=jobs[tile.x];
    const uint i=tile.y*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device uint *value=(device uint *)(bytes+job.ByteOffset)+ulong(i)*job.Stride;
    const uint before=*value;
    if (before==InvalidOffset) return;
    const uint handle=before+job.SourceOrigin, block=handle/MeshElementBlockSize;
    uint slot=WorkHash(block,job.MapCapacity), mapped=InvalidOffset;
    if (job.MapCapacity) while (maps[job.MapOffset+slot].x) {
        const uint2 entry=maps[job.MapOffset+slot];
        if (entry.x==block+1u) { mapped=entry.y*MeshElementBlockSize+handle%MeshElementBlockSize-job.DestinationOrigin; break; }
        slot=(slot+1u)&(job.MapCapacity-1u);
    }
    if (job.Span) value[1]=mapped==InvalidOffset ? InvalidOffset : mapped+(value[1]-before);
    *value=mapped;
}
kernel void RebaseByRank(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], device uchar *bytes [[buffer(CloneBufferIndex_Data)]],
    device const RankRebase *jobs [[buffer(CloneBufferIndex_Jobs)]], device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const RankRebase job=jobs[tile.x];
    const uint i=tile.y*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device uint *handle=(device uint *)(bytes+job.ByteOffset)+ulong(i)*job.Stride;
    const uint rank=MeshletIndexRank(b,job.Index,*handle);
    *handle=rank==InvalidOffset?InvalidOffset:job.First+rank;
}
kernel void GatherByRank(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], device uchar *bytes [[buffer(CloneBufferIndex_Data)]],
    device const RankGather *jobs [[buffer(CloneBufferIndex_Jobs)]], device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const RankGather job=jobs[tile.x];
    const uint i=tile.y*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device const uint *from=(device const uint *)(bytes+ulong(MeshletIndexSelect(b,job.Index,i))*job.Bytes);
    device uint *to=(device uint *)(bytes+ulong(job.Destination+i)*job.Bytes);
    for (uint w=0u; w<job.Bytes/4u; ++w) to[w]=from[w];
}
