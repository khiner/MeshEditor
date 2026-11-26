#pragma once
// Manifold's level set as a topology operator: the source vertices are the lattice points in lattice order, each holding its field value in the job's first parameter attribute.
#include "Lattice.metal"
#include "MeshTopologyContext.metal"

inline auto TopoLattice(TopoContext ctx, device const MeshTopologyJob &job) {
    const auto values = [ctx, &job](uint v) { return TopoParam0Vertex(ctx, job, v); };
    return LatticeField<decltype(values)>{int3(int(job.LatticeSize[0]), int(job.LatticeSize[1]), int(job.LatticeSize[2])), values, job.Param1};
}
inline uint3 TopoLevelSetOutputs(TopoContext ctx, device const MeshTopologyJob &job, uint v) {
    const auto field = TopoLattice(ctx, job);
    const uint2 counts = LatticeCounts(field, LatticeAt(field.Size, v));
    return uint3(counts.x, counts.y, 3u * counts.y);
}
inline void TopoLevelSetEmit(TopoContext ctx, device const MeshTopologyJob &job, uint v, uint entry) {
    const auto field = TopoLattice(ctx, job);
    const LatticeIndex i = LatticeAt(field.Size, v);
    const float d0 = LatticeValue(field, i);
    uint d = ctx.Counts(job, TopoCountVertices)[entry];
    for (uint j = 0u; j < 7u; ++j) {
        const LatticeIndex n = LatticeNeighbor(i, j);
        const float d1 = LatticeValue(field, n);
        if (LatticeInside(d0) == LatticeInside(d1)) continue;
        const float t = d0 == 0.f ? 0.f : d1 == 0.f ? 1.f : d0 / (d0 - d1);
        ctx.WriteVertexMap(job, d, v, LatticeOrdinal(field.Size, n), t);
        ctx.SelectDstVertex(job, d);
        ++d;
    }
    uint fd = ctx.Counts(job, TopoCountFaces)[entry], corner = ctx.Counts(job, TopoCountCorners)[entry];
    LatticeMarch(field, i, [ctx, &job](uint p) { return ctx.Counts(job, TopoCountVertices)[ctx.VertexEntry(p)]; }, [&](uint a, uint b, uint c) {
        const uint corners[3] = {a, b, c};
        for (uint k = 0u; k < 3u; ++k) ctx.WriteCorner(job, corner + k, corners[k], InvalidOffset, InvalidOffset, 0.f, InvalidOffset, true);
        TopoEmitFace(ctx, job, fd, corner, 3u, InvalidOffset, true);
        ++fd;
        corner += 3u;
    });
}
