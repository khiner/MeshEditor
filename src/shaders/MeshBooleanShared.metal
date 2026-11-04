#pragma once
#include "Bindless.metal"
#include "BooleanShadows.metal"
#include "OrderedFloat.metal"
#include "gpu/MeshBoolean.h"

inline uint BoolVertexCount(BooleanJob job) { return job.Vertices[0] + job.Vertices[1]; }
inline uint BoolFaceCount(BooleanJob job) { return job.Faces[0] + job.Faces[1]; }
inline uint BoolTriangleCount(BooleanJob job) { return job.Triangles[0] + job.Triangles[1]; }

constant uint BoolRealOut = 0x80000000u; // A polygon triangle corner whose side to the next corner is its halfedge

// P's vertices shift by a different amount per axis so a slope of one cannot cancel their move.
constant float3 BoolShiftP = float3(1.f / 61.f, 1.f / 89.f, 1.f / 113.f);

// P's vertices move along their displacements by ScaleP plus BoolShiftP, and Q's along their displacements.
// A move is the first-order change of a position, kept apart from it so ties compare moves at full precision.
struct BoolGeometry {
    device const packed_float3 *Positions, *Displacements;
    device const packed_uint3 *Triangles;
    device const uint *Pairs;
    uint FirstQ;
    float Tol, ScaleP;

    float3 Position(uint v) const { return float3(Positions[v]); }
    float3 Move(uint v) const {
        const bool p = v < FirstQ;
        const float3 d = float3(Displacements[v]);
        return p ? d * ScaleP + BoolShiftP : d;
    }
    uint Start(uint h) const { return uint3(Triangles[h / 3u])[h % 3u]; }
    uint End(uint h) const { return uint3(Triangles[h / 3u])[(h % 3u + 1u) % 3u]; }
    uint Pair(uint h) const { return Pairs[h]; }
    bool Forward(uint h) const { return Start(h) < End(h); }
};
inline BoolGeometry Geometry(JobRuns ctx, BooleanJob job) {
    device const uint *bounds = ctx.Run<uint>(job.Bounds);
    float scale = INFINITY;
    for (uint side = 0u; side < 2u; ++side) {
        float largest = 0.f;
        for (uint axis = 0u; axis < 3u; ++axis) largest = max(largest, max(abs(OrderedFloat(bounds[6u * side + axis])), abs(OrderedFloat(bounds[6u * side + 3u + axis]))));
        scale = min(scale, largest);
    }
    const float tol = BoolTolerance(scale);
    // P moves an eighth further than Q, so where faces of P and Q coincide P's is the outer one.
    return {ctx.Run<packed_float3>(job.Positions), ctx.Run<packed_float3>(job.Displacements), ctx.Run<packed_uint3>(job.TriangleVertices), ctx.Run<uint>(job.Pairs),
        job.Vertices[0], tol, job.Operation == BooleanOperation::Union ? 1.125f : -1.125f};
}

inline uint BoolRecordHalfedge(JobRuns ctx, BooleanJob job, uint k) { return ctx.Run<uint>(job.IntersectionRecords)[6u * k]; }
inline uint BoolRecordTriangle(JobRuns ctx, BooleanJob job, uint k) { return ctx.Run<uint>(job.IntersectionRecords)[6u * k + 1u]; }
inline int BoolRecordCrossing(JobRuns ctx, BooleanJob job, uint k) { return as_type<int>(ctx.Run<uint>(job.IntersectionRecords)[6u * k + 2u]); }
inline float3 BoolRecordPosition(JobRuns ctx, BooleanJob job, uint k) {
    device const uint *record = ctx.Run<uint>(job.IntersectionRecords) + 6u * k;
    return float3(as_type<float>(record[3]), as_type<float>(record[4]), as_type<float>(record[5]));
}

inline uint BoolCount(JobRuns ctx, BooleanJob job, uint e) { return uint(abs(ctx.Run<int>(job.Inclusion)[e])); }

inline uint BoolPolygonFirst(JobRuns ctx, BooleanJob job, uint t) { return ctx.Run<uint>(job.Counts)[t]; }
inline uint BoolPartialFirst(JobRuns ctx, BooleanJob job, uint h) { return ctx.Run<uint>(job.Counts)[BoolTriangleCount(job) + h] - job.PolygonHalfedges; }
inline uint BoolGroupFirst(JobRuns ctx, BooleanJob job, uint member) { return ctx.Run<uint>(job.Counts)[4u * BoolTriangleCount(job) + member] - job.PolygonHalfedges - job.PartialEntries; }
inline uint BoolOutputFirst(JobRuns ctx, BooleanJob job, uint e) {
    return ctx.Run<uint>(job.Counts)[4u * BoolTriangleCount(job) + 2u * job.Intersections + e] - job.PolygonHalfedges - job.PartialEntries - job.GroupEntries;
}
inline float3 BoolEntryPosition(JobRuns ctx, BooleanJob job, uint e) {
    const uint vertices = BoolVertexCount(job);
    return e < vertices ? float3(ctx.Run<packed_float3>(job.Positions)[e]) : BoolRecordPosition(ctx, job, e - vertices);
}
inline float3 BoolOutputPosition(JobRuns ctx, BooleanJob job, uint d) { return BoolEntryPosition(ctx, job, ctx.Run<uint>(job.OutputEntries)[d]); }
