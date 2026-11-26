#pragma once
// Manifold's level set: marching tetrahedra over a lattice of grid points and body centers.
// Lattice points have ordinals in lattice order, grid points before body centers, and a field value each.
// Each point owns the seven lattice edges to its first seven neighbors and the six tetrahedra around its edge toward the next body center or grid point.
#include "Bindless.metal"

// A lattice point: its grid index and W, 1 at a grid point and 0 at the body center half a step below it.
struct LatticeIndex {
    int3 P;
    int W;
};
constant int4 LatticeOffsets[14] = {int4(0, 0, 0, 1), int4(1, 0, 0, 0), int4(0, 1, 0, 0), int4(0, 0, 1, 0), int4(-1, 0, 0, 1), int4(0, -1, 0, 1), int4(0, 0, -1, 1), int4(-1, -1, -1, 1), int4(-1, 0, 0, 0), int4(0, -1, 0, 0), int4(0, 0, -1, 0), int4(0, -1, -1, 1), int4(-1, 0, -1, 1), int4(-1, -1, 0, 1)};
// The triangles of a tetrahedron by the inside bits of its corners, as indices of its six edge crossings.
constant int3 LatticeTetTri0[16] = {int3(-1), int3(0, 3, 4), int3(0, 1, 5), int3(1, 5, 3), int3(1, 4, 2), int3(1, 0, 3), int3(2, 5, 0), int3(5, 3, 2), int3(2, 3, 5), int3(0, 5, 2), int3(3, 0, 1), int3(2, 4, 1), int3(3, 5, 1), int3(5, 1, 0), int3(4, 3, 0), int3(-1)};
constant int3 LatticeTetTri1[16] = {int3(-1), int3(-1), int3(-1), int3(3, 4, 1), int3(-1), int3(3, 2, 1), int3(0, 4, 2), int3(-1), int3(-1), int3(2, 4, 0), int3(1, 2, 3), int3(-1), int3(1, 4, 3), int3(-1), int3(-1), int3(-1)};

inline LatticeIndex LatticeNeighbor(LatticeIndex base, uint i) {
    const int4 o = LatticeOffsets[i];
    LatticeIndex n{base.P + o.xyz, base.W + o.w};
    if (n.W == 2) {
        n.P += 1;
        n.W = 0;
    }
    return n;
}
// The ordinal of a lattice point of a lattice of `size` grid points per axis, or InvalidOffset outside the lattice.
inline uint LatticeOrdinal(int3 size, LatticeIndex i) {
    const int3 limit = size + (i.W == 1 ? 0 : 1);
    if (any(i.P < 0) || any(i.P >= limit)) return InvalidOffset;
    const uint base = i.W == 1 ? 0u : uint(size.x * size.y * size.z);
    return base + (uint(i.P.x) * uint(limit.y) + uint(i.P.y)) * uint(limit.z) + uint(i.P.z);
}
inline LatticeIndex LatticeAt(int3 size, uint v) {
    const uint grid = uint(size.x * size.y * size.z);
    const int w = v < grid ? 1 : 0;
    const uint u = w ? v : v - grid;
    const int3 limit = size + (w ? 0 : 1);
    return {int3(int(u / uint(limit.y * limit.z)), int((u / uint(limit.z)) % uint(limit.y)), int(u % uint(limit.z))), w};
}
// A lattice of `Size` grid points per axis whose point of ordinal v holds the field value Value(v), and the level its surface crosses.
template<typename Values> struct LatticeField {
    int3 Size;
    Values Value;
    float Level;
};
// The value at a lattice point less the level, zero outside the lattice and at most zero on the box boundary, which closes the surface at the box.
template<typename Values> inline float LatticeValue(LatticeField<Values> field, LatticeIndex i) {
    const uint v = LatticeOrdinal(field.Size, i);
    if (v == InvalidOffset) return 0.f;
    const float d = field.Value(v) - field.Level;
    const int3 size = field.Size;
    const int lower = min(min(i.P.x, i.P.y), i.P.z), upper = min(min(size.x - i.P.x, size.y - i.P.y), size.z - i.P.z);
    return min(lower, upper - i.W) == 0 ? min(d, 0.f) : d;
}
inline bool LatticeInside(float d) { return d > 0.f; }

// The crossings on a point's seven owned edges, InvalidOffset where the ends agree.
// A crossing is its output vertex once the output is counted, and a key unique to the edge before it.
struct LatticeEdges {
    uint E[7];
};
// `first(v)` is the output vertex of the first crossing of point v once the output is counted, or InvalidOffset to key the crossings.
template<typename Values, typename First> inline LatticeEdges LatticeCrossings(LatticeField<Values> field, LatticeIndex i, First first) {
    LatticeEdges edges;
    const uint v = LatticeOrdinal(field.Size, i);
    if (v == InvalidOffset) {
        for (uint j = 0u; j < 7u; ++j) edges.E[j] = InvalidOffset;
        return edges;
    }
    const float d0 = LatticeValue(field, i);
    const bool keyed = first(v) == InvalidOffset;
    uint next = keyed ? 7u * v : first(v);
    for (uint j = 0u; j < 7u; ++j) {
        const bool crossed = LatticeInside(d0) != LatticeInside(LatticeValue(field, LatticeNeighbor(i, j)));
        edges.E[j] = crossed ? (keyed ? 7u * v + j : next++) : InvalidOffset;
    }
    return edges;
}
// Keys the crossings of a lattice whose output is not counted.
inline uint LatticeKeys(uint) { return InvalidOffset; }
template<typename Values, typename First, typename Emit> inline void LatticeMarch(LatticeField<Values> field, LatticeIndex base, First first, Emit emit) {
    const LatticeEdges base_edges = LatticeCrossings(field, base, first);
    const int base_sign = LatticeInside(LatticeValue(field, base)) ? 1 : -1;
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
    LatticeEdges this_edges = LatticeCrossings(field, this_index, first);
    tet[2] = neighbor_sign(1u);
    for (uint i = 0u; i < 3u; ++i) {
        const uint next3 = (i + 1u) % 3u, prev3 = (i + 2u) % 3u;
        this_index = lead;
        this_index.P[prev3] -= 1;
        const LatticeEdges next_edges = LatticeCrossings(field, this_index, first);
        tet[3] = neighbor_sign(prev3 + 4u);
        const uint edges1[6] = {base_edges.E[0], base_edges.E[i + 1u], next_edges.E[next3 + 4u], next_edges.E[prev3 + 1u], this_edges.E[i + 4u], base_edges.E[prev3 + 4u]};
        this_edges = next_edges;
        create_tet(tet, edges1);
        this_index = base;
        this_index.P[next3] += 1;
        const LatticeEdges following = LatticeCrossings(field, this_index, first);
        tet[2] = tet[3];
        tet[3] = neighbor_sign(next3 + 1u);
        const uint edges2[6] = {base_edges.E[0], edges1[5], this_edges.E[i + 4u], following.E[next3 + 4u], edges1[3], base_edges.E[next3 + 1u]};
        this_edges = following;
        create_tet(tet, edges2);
        tet[2] = tet[3];
    }
}
// The crossings on a point's owned edges and the triangles of its tetrahedra.
template<typename Values> inline uint2 LatticeCounts(LatticeField<Values> field, LatticeIndex i) {
    const LatticeEdges edges = LatticeCrossings(field, i, LatticeKeys);
    uint crossings = 0u, triangles = 0u;
    for (uint j = 0u; j < 7u; ++j) crossings += edges.E[j] != InvalidOffset ? 1u : 0u;
    LatticeMarch(field, i, LatticeKeys, [&](uint, uint, uint) { ++triangles; });
    return uint2(crossings, triangles);
}
