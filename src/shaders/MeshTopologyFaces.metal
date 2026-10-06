#ifndef MESHTOPOLOGYFACES_MSL
#define MESHTOPOLOGYFACES_MSL

// Face-splitting and face-joining rules: triangulation by ear clipping and quad beauty, quad joins, pokes, flips, edge-split sectors, and insets.
#include "MeshTopologyContext.metal"
#include "gpu/PolygonTriangulation.h"

constant float JoinAngleThreshold = 40.f * 3.14159265f / 180.f; // Blender's default face and shape thresholds

inline float2 TopoProject(float3 p, float3 u, float3 v) { return float2(dot(p, u), dot(p, v)); }

inline void TopoPlaneFrame(float3 n, thread float3 &u, thread float3 &v) {
    const float3 axis = abs(n.x) < 0.6f ? float3(1.f, 0.f, 0.f) : abs(n.y) < 0.6f ? float3(0.f, 1.f, 0.f) : float3(0.f, 0.f, 1.f);
    u = normalize(cross(n, axis));
    v = cross(n, u);
}

inline float TopoCross2(float2 a, float2 b, float2 c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }

// Blender's quad beauty: the diagonal whose two triangles keep the larger minimum area-to-perimeter ratio, avoiding flipped or empty triangles.
// Returns true to split along corners 1 and 3 rather than 0 and 2.
inline bool TopoSplitQuad13(float2 p0, float2 p1, float2 p2, float2 p3) {
    const float area_123 = TopoCross2(p1, p2, p3), area_130 = TopoCross2(p1, p3, p0);
    const float area_012 = TopoCross2(p0, p1, p2), area_023 = TopoCross2(p0, p2, p3);
    const float eps = 1e-12f;
    const bool ok_13 = (area_123 >= 0.f) == (area_130 >= 0.f) && abs(area_123) > eps && abs(area_130) > eps;
    const bool ok_02 = (area_012 >= 0.f) == (area_023 >= 0.f) && abs(area_012) > eps && abs(area_023) > eps;
    if (!ok_02) return ok_13;
    if (!ok_13) return false;
    const float l01 = distance(p0, p1), l12 = distance(p1, p2), l23 = distance(p2, p3), l30 = distance(p3, p0);
    const float l02 = distance(p0, p2), l13 = distance(p1, p3);
    const float fac_02 = min(abs(area_012) / (l01 + l12 + l02), abs(area_023) / (l02 + l23 + l30));
    const float fac_13 = min(abs(area_123) / (l12 + l23 + l13), abs(area_130) / (l13 + l30 + l01));
    return fac_13 > fac_02;
}

template<typename Emit>
inline void TopoTriangulateFace(TopoContext ctx, MeshTopologyJob job, uint f, Emit emit) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const auto corners = ctx.SrcCorners(job);
    float3 u, v;
    TopoPlaneFrame(normalize(float3(ctx.SrcFaceNormals(job)[f])), u, v);
    if (n == 4u) {
        float2 p[4];
        for (uint k = 0u; k < 4u; ++k) p[k] = TopoProject(ctx.SrcPosition(job, corners[range.x + k]), u, v);
        if (TopoSplitQuad13(p[0], p[1], p[2], p[3])) {
            emit(uint3(1u, 2u, 3u), 0u);
            emit(uint3(1u, 3u, 0u), 1u);
        } else {
            emit(uint3(0u, 1u, 2u), 0u);
            emit(uint3(0u, 2u, 3u), 1u);
        }
        return;
    }
    const uint first = ctx.SrcHalfedgeDomain(job).Index(range.x);
    device uint *scratch = ctx.Scratch() + job.HalfedgeAuxOffset;
    device uint *next = scratch + first;
    device uint *prev = scratch + job.SrcHalfedgeCount + first;
    device float2 *points = reinterpret_cast<device float2 *>(scratch + 2u * job.SrcHalfedgeCount) + first;
    for (uint k = 0u; k < n; ++k) points[k] = TopoProject(ctx.SrcPosition(job, corners[range.x + k]), u, v);
    PolygonEarClip(points, next, prev, n, emit);
}

// The join cost of the quad a triangle would form with its neighbor across halfedge `h`, or a negative value when the pair does not join.
// Follows Blender's error: normal difference across both diagonals plus each corner's deviation from a right angle.
inline float TopoJoinCost(TopoContext ctx, MeshTopologyJob job, uint h) {
    const uint opposite = ctx.SrcOpposite(job, h);
    if (opposite == InvalidOffset) return -1.f;
    const uint f = ctx.SrcFaceOf(job, h), g = ctx.SrcFaceOf(job, opposite);
    if (f == InvalidOffset || !ctx.SrcSelectedFace(job, g)) return -1.f;
    const uint2 fr = ctx.SrcFaceRange(job, f), gr = ctx.SrcFaceRange(job, g);
    if (fr.y - fr.x != 3u || gr.y - gr.x != 3u) return -1.f;
    const auto corners = ctx.SrcCorners(job);
    const uint q = corners[ctx.SrcPrev(job, h)], r = corners[h];
    const uint p = corners[ctx.SrcNext(job, h)], s = corners[ctx.SrcNext(job, opposite)];
    const float3 P = ctx.SrcPosition(job, p), Q = ctx.SrcPosition(job, q), R = ctx.SrcPosition(job, r), S = ctx.SrcPosition(job, s);
    const float3 nf = normalize(float3(ctx.SrcFaceNormals(job)[f])), ng = normalize(float3(ctx.SrcFaceNormals(job)[g]));
    if (dot(nf, ng) < cos(JoinAngleThreshold)) return -1.f;
    // The quad runs p, q, s, r in the first triangle's winding.
    const float3 e0 = normalize(P - Q), e1 = normalize(Q - S), e2 = normalize(S - R), e3 = normalize(R - P);
    const float a0 = acos(clamp(dot(e0, e1), -1.f, 1.f)), a1 = acos(clamp(dot(e1, e2), -1.f, 1.f));
    const float a2 = acos(clamp(dot(e2, e3), -1.f, 1.f)), a3 = acos(clamp(dot(e3, e0), -1.f, 1.f));
    const float half_pi = 1.57079633f;
    if (max(max(abs(a0 - half_pi), abs(a1 - half_pi)), max(abs(a2 - half_pi), abs(a3 - half_pi))) > JoinAngleThreshold) return -1.f;
    // A quad whose two triangulations disagree in orientation is folded.
    const float3 n_pqs = cross(Q - P, S - P), n_psr = cross(S - P, R - P), n_qsr = cross(S - Q, R - Q), n_qrp = cross(R - Q, P - Q);
    if (dot(n_pqs, n_psr) < 0.f || dot(n_qsr, n_qrp) < 0.f) return -1.f;
    const float angle_a = acos(clamp(dot(normalize(n_pqs), normalize(n_psr)), -1.f, 1.f));
    const float angle_b = acos(clamp(dot(normalize(n_qsr), normalize(n_qrp)), -1.f, 1.f));
    const float normal_error = (angle_a + angle_b) / (2.f * 3.14159265f);
    const float shape_error = (abs(a0 - half_pi) + abs(a1 - half_pi) + abs(a2 - half_pi) + abs(a3 - half_pi)) / (2.f * 3.14159265f);
    return normal_error + shape_error;
}

// The lowest halfedge arriving at corner `h`'s vertex reachable without crossing a selected edge or a boundary: its edge-split sector.
inline uint TopoSectorRep(TopoContext ctx, MeshTopologyJob job, uint h) {
    uint rep = h;
    const uint v = ctx.SrcCorners(job)[h];
    const uint fan_length = ctx.SrcFan(job, v).y;
    // Forward: across the edge leaving the vertex in this face.
    uint current = h;
    for (uint step = 0u; step < fan_length; ++step) {
        const uint out = ctx.SrcNext(job, current);
        if (ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, out))) break;
        const uint opposite = ctx.SrcOpposite(job, out);
        if (opposite == InvalidOffset || opposite == h) break;
        current = opposite;
        rep = min(rep, current);
    }
    // Backward: across this corner's own arriving edge.
    current = h;
    for (uint step = 0u; step < fan_length; ++step) {
        if (ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, current))) break;
        const uint opposite = ctx.SrcOpposite(job, current);
        if (opposite == InvalidOffset) break;
        current = ctx.SrcPrev(job, opposite);
        if (current == h) break;
        rep = min(rep, current);
    }
    return rep;
}

// The output vertex of corner `h` after an edge split: the vertex's own output plus its sector's rank among the sectors at the vertex.
inline uint TopoSectorVertex(TopoContext ctx, MeshTopologyJob job, uint h) {
    const uint v = ctx.SrcCorners(job)[h];
    // Faces in the affected closure include boundary vertices whose other incident faces stay untouched.
    // They keep their original identity.
    // Only endpoints of selected edges need their complete sector fan examined.
    if (!(ctx.FlagVertices(job)[v] & TopoTagged)) return ctx.Counts(job, TopoCountVertices)[v];
    const uint rep = ctx.HalfedgeAux(job)[h];
    const uint2 fan = ctx.SrcFan(job, v);
    // Every sector's corners share one representative, so lower representatives count the sectors ranked below.
    uint sectors_below = 0u;
    for (uint k = 0u; k < fan.y; ++k) {
        const uint corner = ctx.SrcFanCorner(job,fan.x+k);
        if (ctx.HalfedgeAux(job)[corner] == corner && corner < rep) ++sectors_below;
    }
    return ctx.Counts(job, TopoCountVertices)[v] + sectors_below;
}

// How many sectors an edge split leaves at vertex `v`: one per representative corner, or one for a vertex without corners.
inline uint TopoSectorCount(TopoContext ctx, MeshTopologyJob job, uint v) {
    if (!(ctx.FlagVertices(job)[v] & TopoTagged)) return 1u;
    const uint2 fan = ctx.SrcFan(job, v);
    if (fan.y == 0u) return 1u;
    uint sectors = 0u;
    for (uint k = 0u; k < fan.y; ++k) {
        const uint corner = ctx.SrcFanCorner(job,fan.x+k);
        if (ctx.HalfedgeAux(job)[corner] == corner) ++sectors;
    }
    return sectors;
}

// The inset displacement at a corner between edges arriving from `a` and leaving to `b`, inside a face with normal `n`.
inline float3 TopoInsetCorner(float3 a, float3 p, float3 b, float3 n, float thickness, float depth, bool even) {
    const float3 in_prev = normalize(cross(n, p - a)), in_next = normalize(cross(n, b - p));
    const float3 sum = in_prev + in_next;
    const float scale = even ? 1.f / max(1.f + dot(in_prev, in_next), 1e-4f) : 0.5f;
    return sum * (thickness * scale) + n * depth;
}

#endif
