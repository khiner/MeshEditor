#ifndef TOPOLOGY_COLLAPSE_SHARED_MSL
#define TOPOLOGY_COLLAPSE_SHARED_MSL

#include "MeshTopologyContext.metal"
#include "RadixSort.metal"

struct CollapseSum { packed_float3 Delta; uint Count; };

inline uint CollapseLevelCount(uint count, uint level) {
    for (uint l = 0u; l < level; ++l) {
        if (count <= 256u) return 0u;
        count = count / 256u + uint(count % 256u != 0u);
    }
    return count;
}
inline uint CollapseLevelOffset(uint count, uint level) {
    uint offset = 0u;
    for (uint l = 0u; l < level; ++l) {
        offset += count;
        count = count / 256u + uint(count % 256u != 0u);
    }
    return offset;
}
inline RadixSortView CollapseSort(TopoContext ctx, MeshTopologyJob job, uint shift = 0u) {
    device uint *base = ctx.Scratch() + job.CollapseOffset;
    const uint n = job.CollapseCount, blocks = n / 256u + uint(n % 256u != 0u);
    return {base, base + n, base + 2u * n, base + 3u * n, base + 3u * n + 16u * blocks,
            n, blocks, 1u, 0u, shift, ((shift / 4u) & 1u) != 0u};
}
inline device CollapseSum *CollapseSums(TopoContext ctx, MeshTopologyJob job, uint level) {
    const auto sort = CollapseSort(ctx, job);
    return reinterpret_cast<device CollapseSum *>(sort.Totals + 16u) + CollapseLevelOffset(job.CollapseCount, level);
}
inline uint CollapseVertex(TopoContext ctx, MeshTopologyJob job, uint rank) {
    return job.CollapseVertices.Slot == InvalidSlot ? rank :
        ctx.SrcVertexDomain(job).Index(BindlessBuffer(uint, ctx.B.Buffer, job.CollapseVertices.Slot)[job.CollapseVertices.Offset + rank]);
}
inline uint CollapseKey(RadixSortView sort, uint i, uint level) {
    const ulong end = min((ulong(i) + 1u) << (8u * level), ulong(sort.Count));
    return sort.Keys[sort.Order[uint(end - 1u)]];
}
inline CollapseSum AddCollapseSum(CollapseSum a, CollapseSum b) {
    return {packed_float3(float3(a.Delta) + float3(b.Delta)), a.Count + b.Count};
}

#endif
