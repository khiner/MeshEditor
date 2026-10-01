#include <metal_stdlib>
using namespace metal;
kernel void RebaseIndices(device uint *indices [[buffer(0)]], constant uint3 &pc [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < pc.x && indices[ulong(i) * pc.z] != 0xffffffffu) indices[ulong(i) * pc.z] += pc.y;
}
kernel void CopyReferencePairs(device packed_uint2 *values [[buffer(0)]], device const packed_uint3 *jobs [[buffer(1)]],
    device const uint2 *tiles [[buffer(2)]], constant uint2 &delta [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint2 tile=tiles[group]; const uint3 job=uint3(jobs[tile.x]);
    const uint i=tile.y+lane;
    if (i>=job.z) return;
    const uint2 before=uint2(values[job.x+i]);
    values[job.y+i]=packed_uint2(select(before+delta,before,before==uint2(0xffffffffu)));
}
