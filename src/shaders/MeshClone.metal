#include "gpu/MeshCloneJob.h"
// Copies each run sixteen bytes per thread, a whole vector when the run is aligned to it.
kernel void CloneCopyByteRuns(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const ByteCopy *jobs [[buffer(CloneBufferIndex_Jobs)]], device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]],
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
kernel void CloneOffsetHandles(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const HandleOffset *jobs [[buffer(CloneBufferIndex_Jobs)]],
    device const uint2 *tiles [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const HandleOffset job=jobs[tile.x];
    const uint i=tile.y*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device uint *value=(device uint *)(bytes+job.ByteOffset)+ulong(i)*job.Stride;
    if (*value!=InvalidOffset) *value+=job.Shift;
}
