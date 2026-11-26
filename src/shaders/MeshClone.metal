#include "TileJob.metal"
#include "gpu/MeshCloneJob.h"
// Copies each run sixteen bytes per thread, a whole vector when the run is aligned to it.
kernel void CloneCopyByteRuns(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const ByteCopy *jobs [[buffer(CloneBufferIndex_Jobs)]], device const uint *firsts [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint tile;
    const ByteCopy job=jobs[TileJob(firsts,group,tile)];
    const uint tiles=uint((job.Bytes+CloneCopyTileBytes-1u)/CloneCopyTileBytes);
    const ulong first=ulong(tile%tiles)*CloneCopyTileBytes+ulong(lane)*CloneCopyThreadBytes;
    if (first>=job.Bytes) return;
    const ulong destination=job.Destination+ulong(tile/tiles)*job.Step;
    if (((job.Source|destination|job.Bytes)&ulong(CloneCopyThreadBytes-1u))==0ul) {
        *(device uint4 *)(bytes+destination+first)=*(device const uint4 *)(bytes+job.Source+first);
        return;
    }
    for (ulong i=first, end=min(first+CloneCopyThreadBytes,job.Bytes); i<end; ++i) bytes[destination+i]=bytes[job.Source+i];
}
kernel void CloneOffsetHandles(device uchar *bytes [[buffer(CloneBufferIndex_Data)]], device const HandleOffset *jobs [[buffer(CloneBufferIndex_Jobs)]],
    device const uint *firsts [[buffer(CloneBufferIndex_Tiles)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint tile;
    const HandleOffset job=jobs[TileJob(firsts,group,tile)];
    const uint tiles=(job.Count+CloneRunThreads-1u)/CloneRunThreads, copy=tile/tiles;
    const uint i=(tile%tiles)*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device uint *value=(device uint *)(bytes+job.ByteOffset+ulong(copy)*job.Step)+ulong(i)*job.Stride;
    if (*value!=InvalidOffset) *value+=job.Shift+copy*job.ShiftStep;
}
