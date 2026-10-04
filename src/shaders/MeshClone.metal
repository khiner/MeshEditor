#include "gpu/MeshCloneJob.h"
// Copies each run sixteen bytes per thread, a whole vector when the run is aligned to it.
kernel void CopyByteRuns(device uchar *bytes [[buffer(0)]], device const ByteCopy *jobs [[buffer(1)]], device const uint2 *tiles [[buffer(2)]],
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
kernel void RebaseIndexRuns(device uchar *bytes [[buffer(0)]], device const IndexRebase *jobs [[buffer(1)]], device const uint2 *tiles [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const IndexRebase job=jobs[tile.x];
    const uint i=tile.y*CloneRunThreads+lane;
    if (i>=job.Count) return;
    device uint *index=(device uint *)(bytes+job.ByteOffset)+ulong(i)*job.Stride;
    if (*index!=0xffffffffu) *index+=job.Delta;
}
kernel void CopyReferencePairs(device packed_uint2 *values [[buffer(0)]], device const ReferencePairCopy *jobs [[buffer(1)]],
    device const uint2 *tiles [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const ReferencePairCopy job=jobs[tile.x];
    const uint i=tile.y*ClonePairThreads+lane;
    if (i>=job.Count) return;
    const uint2 before=uint2(values[job.Source+i]);
    values[job.Destination+i]=packed_uint2(select(before+uint2(job.FirstDelta,job.SecondDelta),before,before==uint2(0xffffffffu)));
}
