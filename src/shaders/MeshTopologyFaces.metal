#pragma once
#include "MeshTopologyContext.metal"
#include "gpu/PolygonTriangulation.h"

constant float JoinAngleThreshold = 40.f * M_PI_F / 180.f; // Blender's default face and shape thresholds

inline float2 TopoProject(float3 p, float3 u, float3 v) { return float2(dot(p, u), dot(p, v)); }

inline void TopoPlaneFrame(float3 n, thread float3 &u, thread float3 &v) {
    const float3 axis = abs(n.x) < 0.6f ? float3(1.f, 0.f, 0.f) : abs(n.y) < 0.6f ? float3(0.f, 1.f, 0.f) : float3(0.f, 0.f, 1.f);
    u = normalize(cross(n, axis));
    v = cross(n, u);
}

inline float TopoCross2(float2 a, float2 b, float2 c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }

template<typename Emit>
inline void TopoTriangulateFace(TopoContext ctx, MeshTopologyJob job, uint f, Emit emit) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const auto corners = ctx.SrcCorners(job);
    const uint first = ctx.SrcHalfedgeDomain(job).Index(range.x);
    device uint *scratch = ctx.Scratch() + job.HalfedgeAuxOffset;
    TriangulateWithScratch(range.y - range.x, [&](uint k) { return ctx.SrcPosition(job, corners[range.x + k]); },
        scratch + 3u * first, reinterpret_cast<device float *>(scratch + 3u * job.SrcHalfedgeCount) + 3u * first, emit);
}

// The join cost of the quad a triangle would form with its neighbor across halfedge `h`, or a negative value when the pair does not join.
inline float TopoJoinCost(TopoContext ctx, MeshTopologyJob job, uint h) {
    // Follows Blender's error: normal difference across both diagonals plus each corner's deviation from a right angle.
    const uint opposite = ctx.SrcOpposite(job, h);
    if (opposite == InvalidOffset) return -1.f;
    const uint f = ctx.SrcFaceOf(job, h), g = ctx.SrcFaceOf(job, opposite);
    if (f == InvalidOffset || !ctx.SrcSelectedFace(job, g)) return -1.f;
    const uint2 fr = ctx.SrcFaceRange(job, f), gr = ctx.SrcFaceRange(job, g);
    if (fr.y - fr.x != 3u || gr.y - gr.x != 3u) return -1.f;
    const auto corners = ctx.SrcCorners(job);
    const uint q = ctx.SrcFrom(job, h), r = corners[h];
    const uint p = corners[ctx.SrcNext(job, h)], s = corners[ctx.SrcNext(job, opposite)];
    const float3 P = ctx.SrcPosition(job, p), Q = ctx.SrcPosition(job, q), R = ctx.SrcPosition(job, r), S = ctx.SrcPosition(job, s);
    const float3 nf = normalize(float3(ctx.SrcFaceNormals(job)[f])), ng = normalize(float3(ctx.SrcFaceNormals(job)[g]));
    if (dot(nf, ng) < cos(JoinAngleThreshold)) return -1.f;
    const float3 e0 = normalize(P - Q), e1 = normalize(Q - S), e2 = normalize(S - R), e3 = normalize(R - P);
    const float a0 = acos(clamp(dot(e0, e1), -1.f, 1.f)), a1 = acos(clamp(dot(e1, e2), -1.f, 1.f));
    const float a2 = acos(clamp(dot(e2, e3), -1.f, 1.f)), a3 = acos(clamp(dot(e3, e0), -1.f, 1.f));
    if (max(max(abs(a0 - M_PI_2_F), abs(a1 - M_PI_2_F)), max(abs(a2 - M_PI_2_F), abs(a3 - M_PI_2_F))) > JoinAngleThreshold) return -1.f;
    const float3 n_pqs = cross(Q - P, S - P), n_psr = cross(S - P, R - P), n_qsr = cross(S - Q, R - Q), n_qrp = cross(R - Q, P - Q);
    if (dot(n_pqs, n_psr) < 0.f || dot(n_qsr, n_qrp) < 0.f) return -1.f;
    const float angle_a = acos(clamp(dot(normalize(n_pqs), normalize(n_psr)), -1.f, 1.f));
    const float angle_b = acos(clamp(dot(normalize(n_qsr), normalize(n_qrp)), -1.f, 1.f));
    const float normal_error = (angle_a + angle_b) / (2.f * M_PI_F);
    const float shape_error = (abs(a0 - M_PI_2_F) + abs(a1 - M_PI_2_F) + abs(a2 - M_PI_2_F) + abs(a3 - M_PI_2_F)) / (2.f * M_PI_F);
    return normal_error + shape_error;
}

// The lowest halfedge arriving at corner `h`'s vertex reachable without crossing a selected edge or a boundary: its edge-split sector.
inline uint TopoSectorRep(TopoContext ctx, MeshTopologyJob job, uint h) {
    uint rep = h;
    const uint v = ctx.SrcCorners(job)[h];
    const uint fan_length = ctx.SrcFan(job, v).y;
    uint current = h;
    for (uint step = 0u; step < fan_length; ++step) {
        const uint out = ctx.SrcNext(job, current);
        if (ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, out))) break;
        const uint opposite = ctx.SrcOpposite(job, out);
        if (opposite == InvalidOffset || opposite == h) break;
        current = opposite;
        rep = min(rep, current);
    }
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
    if (!(ctx.FlagVertices(job)[v] & TopoTagged)) return ctx.Counts(job, TopoCountVertices)[v];
    const uint rep = ctx.HalfedgeAux(job)[h];
    const uint2 fan = ctx.SrcFan(job, v);
    uint sectors_below = 0u;
    for (uint k = 0u; k < fan.y; ++k) {
        const uint corner = ctx.SrcFanCorner(job,fan.x+k);
        if (ctx.HalfedgeAux(job)[corner] == corner && corner < rep) ++sectors_below;
    }
    return ctx.Counts(job, TopoCountVertices)[v] + sectors_below;
}

// One sector per representative corner at an edge-split vertex, and one for any other vertex.
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
