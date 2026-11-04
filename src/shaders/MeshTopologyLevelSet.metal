#pragma once
// Manifold's level set: marching tetrahedra over a lattice of grid points and body centers.
// The source vertices are the lattice points in lattice order, grid points before body centers.
// Each holds its field value in the job's first parameter attribute.
// Each point owns the seven lattice edges to its first seven neighbors and the six tetrahedra around its edge toward the next body center or grid point.
#include "MeshTopologyContext.metal"

// A lattice point: its grid index and W, 1 at a grid point and 0 at the body center half a step below it.
struct LatticeIndex {
    int3 P;
    int W;
};
constant int4 LatticeOffsets[14] = {int4(0, 0, 0, 1), int4(1, 0, 0, 0), int4(0, 1, 0, 0), int4(0, 0, 1, 0), int4(-1, 0, 0, 1), int4(0, -1, 0, 1), int4(0, 0, -1, 1), int4(-1, -1, -1, 1), int4(-1, 0, 0, 0), int4(0, -1, 0, 0), int4(0, 0, -1, 0), int4(0, -1, -1, 1), int4(-1, 0, -1, 1), int4(-1, -1, 0, 1)};
// The triangles of a tetrahedron by the inside bits of its corners, as indices of its six edge crossings.
constant int3 LatticeTetTri0[16] = {int3(-1), int3(0, 3, 4), int3(0, 1, 5), int3(1, 5, 3), int3(1, 4, 2), int3(1, 0, 3), int3(2, 5, 0), int3(5, 3, 2), int3(2, 3, 5), int3(0, 5, 2), int3(3, 0, 1), int3(2, 4, 1), int3(3, 5, 1), int3(5, 1, 0), int3(4, 3, 0), int3(-1)};
constant int3 LatticeTetTri1[16] = {int3(-1), int3(-1), int3(-1), int3(3, 4, 1), int3(-1), int3(3, 2, 1), int3(0, 4, 2), int3(-1), int3(-1), int3(2, 4, 0), int3(1, 2, 3), int3(-1), int3(1, 4, 3), int3(-1), int3(-1), int3(-1)};

inline int3 LatticeSize(MeshTopologyJob job) { return int3(int(job.LatticeSize[0]), int(job.LatticeSize[1]), int(job.LatticeSize[2])); }
inline LatticeIndex LatticeNeighbor(LatticeIndex base, uint i) {
    const int4 o = LatticeOffsets[i];
    LatticeIndex n{base.P + o.xyz, base.W + o.w};
    if (n.W == 2) {
        n.P += 1;
        n.W = 0;
    }
    return n;
}
// The source vertex of a lattice point, or InvalidOffset outside the lattice.
inline uint LatticeOrdinal(MeshTopologyJob job, LatticeIndex i) {
    const int3 size = LatticeSize(job), limit = size + (i.W == 1 ? 0 : 1);
    if (any(i.P < 0) || any(i.P >= limit)) return InvalidOffset;
    const uint base = i.W == 1 ? 0u : uint(size.x * size.y * size.z);
    return base + (uint(i.P.x) * uint(limit.y) + uint(i.P.y)) * uint(limit.z) + uint(i.P.z);
}
inline LatticeIndex LatticeAt(MeshTopologyJob job, uint v) {
    const int3 size = LatticeSize(job);
    const uint grid = uint(size.x * size.y * size.z);
    const int w = v < grid ? 1 : 0;
    const uint u = w ? v : v - grid;
    const int3 limit = size + (w ? 0 : 1);
    return {int3(int(u / uint(limit.y * limit.z)), int((u / uint(limit.z)) % uint(limit.y)), int(u % uint(limit.z))), w};
}
// The value at a lattice point less the level, zero outside the lattice and at most zero on the box boundary, which closes the surface at the box.
inline float LatticeValue(TopoContext ctx, MeshTopologyJob job, LatticeIndex i) {
    const uint v = LatticeOrdinal(job, i);
    if (v == InvalidOffset) return 0.f;
    const float d = TopoParam0Vertex(ctx, job, v) - job.Param1;
    const int3 size = LatticeSize(job);
    const int lower = min(min(i.P.x, i.P.y), i.P.z), upper = min(min(size.x - i.P.x, size.y - i.P.y), size.z - i.P.z);
    return min(lower, upper - i.W) == 0 ? min(d, 0.f) : d;
}
inline bool LatticeInside(float d) { return d > 0.f; }

// The crossings on a point's seven owned edges, InvalidOffset where the ends agree.
// A crossing is its output vertex after the count scan, and a key unique to the edge before it.
struct LatticeEdges {
    uint E[7];
};
inline LatticeEdges LatticeCrossings(TopoContext ctx, MeshTopologyJob job, LatticeIndex i, bool scanned) {
    LatticeEdges edges;
    const uint v = LatticeOrdinal(job, i);
    if (v == InvalidOffset) {
        for (uint j = 0u; j < 7u; ++j) edges.E[j] = InvalidOffset;
        return edges;
    }
    const float d0 = LatticeValue(ctx, job, i);
    uint next = scanned ? ctx.Counts(job, TopoCountVertices)[ctx.VertexEntry(v)] : 7u * v;
    for (uint j = 0u; j < 7u; ++j) {
        const bool crossed = LatticeInside(d0) != LatticeInside(LatticeValue(ctx, job, LatticeNeighbor(i, j)));
        edges.E[j] = crossed ? (scanned ? next++ : 7u * v + j) : InvalidOffset;
    }
    return edges;
}
template<typename Emit> inline void LatticeMarch(TopoContext ctx, MeshTopologyJob job, LatticeIndex base, bool scanned, Emit emit) {
    const LatticeEdges base_edges = LatticeCrossings(ctx, job, base, scanned);
    const int base_sign = LatticeInside(LatticeValue(ctx, job, base)) ? 1 : -1;
    const auto neighbor_sign = [&](uint e) { return base_sign * (base_edges.E[e] == InvalidOffset ? 1 : -1); };
    const auto create = [&](int3 tri, thread const uint *edges) {
        if (tri.x < 0) return;
        const uint a = edges[tri.x], b = edges[tri.y], c = edges[tri.z];
        if (a == b || b == c || c == a) return;
        emit(a, b, c);
    };
    const auto create_tet = [&](thread const int *tet, thread const uint *edges) {
        const int i = (tet[0] > 0 ? 1 : 0) + (tet[1] > 0 ? 2 : 0) + (tet[2] > 0 ? 4 : 0) + (tet[3] > 0 ? 8 : 0);
        create(LatticeTetTri0[i], edges);
        create(LatticeTetTri1[i], edges);
    };
    const LatticeIndex lead = LatticeNeighbor(base, 0u);
    int tet[4] = {neighbor_sign(0u), base_sign, -2, -2};
    LatticeIndex this_index = base;
    this_index.P.x += 1;
    LatticeEdges this_edges = LatticeCrossings(ctx, job, this_index, scanned);
    tet[2] = neighbor_sign(1u);
    for (uint i = 0u; i < 3u; ++i) {
        const uint next3 = (i + 1u) % 3u, prev3 = (i + 2u) % 3u;
        this_index = lead;
        this_index.P[prev3] -= 1;
        const LatticeEdges next_edges = LatticeCrossings(ctx, job, this_index, scanned);
        tet[3] = neighbor_sign(prev3 + 4u);
        const uint edges1[6] = {base_edges.E[0], base_edges.E[i + 1u], next_edges.E[next3 + 4u], next_edges.E[prev3 + 1u], this_edges.E[i + 4u], base_edges.E[prev3 + 4u]};
        this_edges = next_edges;
        create_tet(tet, edges1);
        this_index = base;
        this_index.P[next3] += 1;
        const LatticeEdges following = LatticeCrossings(ctx, job, this_index, scanned);
        tet[2] = tet[3];
        tet[3] = neighbor_sign(next3 + 1u);
        const uint edges2[6] = {base_edges.E[0], edges1[5], this_edges.E[i + 4u], following.E[next3 + 4u], edges1[3], base_edges.E[next3 + 1u]};
        this_edges = following;
        create_tet(tet, edges2);
        tet[2] = tet[3];
    }
}
inline uint3 TopoLevelSetOutputs(TopoContext ctx, MeshTopologyJob job, uint v) {
    const LatticeIndex i = LatticeAt(job, v);
    const LatticeEdges edges = LatticeCrossings(ctx, job, i, false);
    uint crossings = 0u, triangles = 0u;
    for (uint j = 0u; j < 7u; ++j) crossings += edges.E[j] != InvalidOffset ? 1u : 0u;
    LatticeMarch(ctx, job, i, false, [&](uint, uint, uint) { ++triangles; });
    return uint3(crossings, triangles, 3u * triangles);
}
inline void TopoLevelSetEmit(TopoContext ctx, MeshTopologyJob job, uint v, uint entry) {
    const LatticeIndex i = LatticeAt(job, v);
    const float d0 = LatticeValue(ctx, job, i);
    uint d = ctx.Counts(job, TopoCountVertices)[entry];
    for (uint j = 0u; j < 7u; ++j) {
        const LatticeIndex n = LatticeNeighbor(i, j);
        const float d1 = LatticeValue(ctx, job, n);
        if (LatticeInside(d0) == LatticeInside(d1)) continue;
        const float t = d0 == 0.f ? 0.f : d1 == 0.f ? 1.f : d0 / (d0 - d1);
        ctx.WriteVertexMap(job, d, v, LatticeOrdinal(job, n), t);
        ctx.SelectDstVertex(job, d);
        ++d;
    }
    uint fd = ctx.Counts(job, TopoCountFaces)[entry], corner = ctx.Counts(job, TopoCountCorners)[entry];
    LatticeMarch(ctx, job, i, true, [&](uint a, uint b, uint c) {
        const uint corners[3] = {a, b, c};
        for (uint k = 0u; k < 3u; ++k) ctx.WriteCorner(job, corner + k, corners[k], InvalidOffset, InvalidOffset, 0.f, InvalidOffset, true);
        TopoEmitFace(ctx, job, fd, corner, 3u, InvalidOffset, true);
        ++fd;
        corner += 3u;
    });
}
