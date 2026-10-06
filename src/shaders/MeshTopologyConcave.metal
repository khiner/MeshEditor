#ifndef MESHTOPOLOGYCONCAVE_MSL
#define MESHTOPOLOGYCONCAVE_MSL

#include "MeshTopologyFaceSplit.metal"

// Temporary triangulation of one selected polygon. Internal edge identities
// survive rotations; the indexed heap updates only the five changed costs.
struct ConcaveFace {
    FaceSplitPolygons Plan;
    device uint *Opposite, *Edge, *Representative, *Concave, *Heap, *HeapPosition;
    device float *Cost;
    device uint *Hash;
    float3 U, V, Normal;
    uint HeapCount;

    bool Before(uint a, uint b, bool beauty) const {
        if (!beauty) {
            const uint h = Representative[a], g = Representative[b];
            const bool ac = Concave[Plan.Source[h]] && Concave[Plan.Source[Plan.Next[h]]];
            const bool bc = Concave[Plan.Source[g]] && Concave[Plan.Source[Plan.Next[g]]];
            if (ac != bc) return !ac;
        }
        if (Cost[a] != Cost[b]) return beauty ? Cost[a] < Cost[b] : Cost[a] > Cost[b];
        return a < b;
    }
    void Swap(uint a, uint b) {
        const uint edge = Heap[a]; Heap[a] = Heap[b]; Heap[b] = edge;
        HeapPosition[Heap[a]] = a; HeapPosition[Heap[b]] = b;
    }
    void Down(uint i, bool beauty) {
        while (2u * i + 1u < HeapCount) {
            uint child = 2u * i + 1u;
            if (child + 1u < HeapCount && Before(Heap[child + 1u], Heap[child], beauty)) ++child;
            if (!Before(Heap[child], Heap[i], beauty)) break;
            Swap(i, child); i = child;
        }
    }
    void Set(uint edge, float cost, bool beauty) {
        Cost[edge] = cost;
        uint i = HeapPosition[edge];
        if (i == InvalidOffset) { i = HeapCount++; Heap[i] = edge; HeapPosition[edge] = i; }
        while (i && Before(Heap[i], Heap[(i - 1u) / 2u], beauty)) {
            const uint parent = (i - 1u) / 2u; Swap(i, parent); i = parent;
        }
        Down(i, beauty);
    }
    uint Pop(bool beauty) {
        const uint edge = Heap[0];
        HeapPosition[edge] = InvalidOffset;
        if (--HeapCount) { Heap[0] = Heap[HeapCount]; HeapPosition[Heap[0]] = 0u; Down(0u, beauty); }
        return edge;
    }
    float Beauty(uint edge, bool updated = false) const {
        const uint h = Representative[edge], g = Opposite[h];
        const float2 a = Plan.Project(h, U, V), b = Plan.Project(Plan.Next[h], U, V);
        const float2 c = Plan.Project(Plan.Prev[h], U, V), d = Plan.Project(Plan.Prev[g], U, V);
        const float old_a = TopoCross2(a, b, c), old_b = TopoCross2(b, a, d);
        const float new_a = TopoCross2(c, d, b), new_b = TopoCross2(d, c, a);
        // Reject flipped or empty replacements, as in Blender's polyfill beauty.
        if (new_a <= 1e-12f || new_b <= 1e-12f) return INFINITY;
        if (old_a <= 1e-12f || old_b <= 1e-12f) return -INFINITY;
        const float ab = distance(a, b), cd = distance(c, d);
        const float ac = distance(a, c), bc = distance(b, c), ad = distance(a, d), bd = distance(b, d);
        const float cost = old_a / (ab + ac + bc) + old_b / (ab + ad + bd) -
            (new_a / (cd + bc + bd) + new_b / (cd + ac + ad));
        // Blender filters updated costs by area to stop roundoff-driven cycles.
        const float area = (old_a + old_b + new_a + new_b) * 0.125f;
        return updated && cost >= -1e-6f * max(area, 1.f) ? INFINITY : cost;
    }
    void Side(uint node, uint opposite, uint edge) {
        Opposite[node] = opposite; Edge[node] = edge;
        if (opposite != InvalidOffset) Opposite[opposite] = node;
        if (edge != InvalidOffset) Representative[edge] = node;
    }
    void Rotate(uint edge) {
        const uint h = Representative[edge], g = Opposite[h];
        const uint hb = Plan.Next[h], hc = Plan.Prev[h], ga = Plan.Next[g], gd = Plan.Prev[g];
        const uint4 opposite = uint4(Opposite[hb], Opposite[hc], Opposite[ga], Opposite[gd]);
        const uint4 edges = uint4(Edge[hb], Edge[hc], Edge[ga], Edge[gd]);
        const uint a = Plan.Source[h], b = Plan.Source[hb], c = Plan.Source[hc], d = Plan.Source[gd];
        Plan.Source[h] = c; Plan.Source[hb] = d; Plan.Source[hc] = b;
        Plan.Source[g] = d; Plan.Source[ga] = c; Plan.Source[gd] = a;
        Side(hc, opposite.x, edges.x); Side(ga, opposite.y, edges.y);
        Side(gd, opposite.z, edges.z); Side(hb, opposite.w, edges.w);
        Side(h, g, edge); Side(g, h, edge);
        Set(edge, INFINITY, true);
        for (uint i = 0u; i < 4u; ++i) if (edges[i] != InvalidOffset) Set(edges[i], Beauty(edges[i], true), true);
    }
    uint Root(uint face) {
        // Length is free until the final loops are counted, so it holds parents.
        while (Plan.Length[face] != face) {
            Plan.Length[face] = Plan.Length[Plan.Length[face]];
            face = Plan.Length[face];
        }
        return face;
    }
    bool Join(uint edge) {
        const uint h = Representative[edge], g = Opposite[h];
        const uint a = Root(h / 3u), b = Root(g / 3u);
        if (a == b) return false;
        // Only the turns at the two endpoints change when the diagonal goes.
        const float3 pa = Plan.Point(h), pb = Plan.Point(g);
        if (dot(cross(pa - Plan.Point(Plan.Prev[h]), Plan.Point(Plan.Next[Plan.Next[g]]) - pa), Normal) <= 1.1920929e-7f ||
            dot(cross(pb - Plan.Point(Plan.Prev[g]), Plan.Point(Plan.Next[Plan.Next[h]]) - pb), Normal) <= 1.1920929e-7f) return false;
        Plan.Next[Plan.Prev[h]] = Plan.Next[g]; Plan.Prev[Plan.Next[g]] = Plan.Prev[h];
        Plan.Next[Plan.Prev[g]] = Plan.Next[h]; Plan.Prev[Plan.Next[h]] = Plan.Prev[g];
        Plan.Length[b] = a;
        Plan.Start[a] = Plan.Next[h];
        return true;
    }
};

// Blender's connect_verts_concave: triangulate, beautify, then join long
// diagonals first, leaving diagonals between original concave corners last.
inline uint3 TopoConcaveCount(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    if (!ctx.SrcSelectedFace(job, f) || n <= 3u) return uint3(0u, 1u, n);
    const auto plan = TopoInitFaceSplit(ctx, job, f);
    const float3 normal = NormalizeOrZero(float3(ctx.SrcFaceNormals(job)[f]));
    device uint *scratch = reinterpret_cast<device uint *>(plan.Positions + n);
    ConcaveFace face{plan, scratch, scratch + 3u * n, scratch + 6u * n, scratch + 7u * n,
        scratch + 8u * n, scratch + 9u * n, reinterpret_cast<device float *>(scratch + 10u * n), scratch + 11u * n,
        float3(0.f), float3(0.f), normal, 0u};
    bool concave = false;
    for (uint i = 0u; i < n; ++i) {
        face.Concave[i] = dot(cross(plan.Point(i) - plan.Point(plan.Prev[i]), plan.Point(plan.Next[i]) - plan.Point(i)), normal) <= 0.f;
        concave = concave || face.Concave[i];
    }
    if (!concave || dot(normal, normal) == 0.f) return uint3(0u, 1u, n);
    TopoPlaneFrame(normal, face.U, face.V);
    TopoTriangulateFace(ctx, job, f, [&](uint3 triangle, uint t) {
        for (uint k = 0u; k < 3u; ++k) {
            const uint h = 3u * t + k;
            plan.Source[h] = triangle[k]; plan.Next[h] = 3u * t + (k + 1u) % 3u; plan.Prev[h] = 3u * t + (k + 2u) % 3u;
            face.Opposite[h] = InvalidOffset; face.Edge[h] = InvalidOffset;
        }
        plan.Start[t] = 3u * t;
        plan.Length[t] = t;
    });
    const uint mask = (1u << (32u - clz(2u * n - 1u))) - 1u;
    for (uint i = 0u; i <= mask; ++i) face.Hash[i] = 0u;
    uint edges = 0u;
    for (uint h = 0u; h < 3u * (n - 2u); ++h) {
        const uint a = plan.Source[h], b = plan.Source[plan.Next[h]], lo = min(a, b), hi = max(a, b);
        if (hi == lo + 1u || (lo == 0u && hi == n - 1u)) continue;
        uint slot = ((lo * 0x9e3779b1u) ^ (hi * 0x85ebca6bu)) & mask;
        while (face.Hash[slot]) {
            const uint g = face.Hash[slot] - 1u;
            if (plan.Source[g] == b && plan.Source[plan.Next[g]] == a) break;
            slot = (slot + 1u) & mask;
        }
        if (!face.Hash[slot]) face.Hash[slot] = h + 1u;
        else {
            const uint g = face.Hash[slot] - 1u;
            face.Side(h, g, edges); face.Side(g, h, edges);
            face.HeapPosition[edges++] = InvalidOffset;
        }
    }
    for (uint e = 0u; e < edges; ++e) face.Set(e, face.Beauty(e), true);
    while (face.HeapCount && face.Cost[face.Heap[0]] < 0.f) face.Rotate(face.Pop(true));
    face.HeapCount = 0u;
    for (uint e = 0u; e < edges; ++e) {
        face.HeapPosition[e] = InvalidOffset;
        const uint h = face.Representative[e];
        const float3 delta = plan.Point(h) - plan.Point(plan.Next[h]);
        face.Set(e, dot(delta, delta), false);
    }
    while (face.HeapCount) face.Join(face.Pop(false));
    uint faces = 0u, corners = 0u;
    for (uint i = 0u; i < n - 2u; ++i) if (plan.Length[i] == i) {
        const uint start = plan.Start[i];
        uint length = 0u, h = start;
        do { ++length; h = plan.Next[h]; } while (h != start);
        plan.Start[faces] = start; plan.Length[faces++] = length;
        corners += length;
    }
    ctx.FlagFaces(job)[f] = faces;
    return uint3(0u, faces, corners);
}
#endif
