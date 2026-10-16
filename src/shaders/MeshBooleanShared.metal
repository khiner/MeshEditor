#ifndef MESHBOOLEANSHARED_MSL
#define MESHBOOLEANSHARED_MSL

// A boolean job's runs, read through any context with Run<T>(offset) over the boolean's scratch storage.
#include "gpu/MeshBoolean.h"

inline uint BoolVertexCount(BooleanJob job) { return job.Vertices[0] + job.Vertices[1]; }
inline uint BoolFaceCount(BooleanJob job) { return job.Faces[0] + job.Faces[1]; }
inline uint BoolTriangleCount(BooleanJob job) { return job.Triangles[0] + job.Triangles[1]; }

// Combined vertices, triangles and halfedges with their positions, perturbation normals, triangle normals and pairs.
struct BoolGeometry {
    device const packed_float3 *Positions, *Normals, *FaceNormals;
    device const packed_uint3 *Triangles;
    device const uint *Pairs;

    float3 Position(uint v) const { return float3(Positions[v]); }
    float3 Normal(uint v) const { return float3(Normals[v]); }
    float3 FaceNormal(uint t) const { return float3(FaceNormals[t]); }
    uint Start(uint h) const { return uint3(Triangles[h / 3u])[h % 3u]; }
    uint End(uint h) const { return uint3(Triangles[h / 3u])[(h % 3u + 1u) % 3u]; }
    uint Pair(uint h) const { return Pairs[h]; }
    bool Forward(uint h) const { return Start(h) < End(h); }
};
template<typename Runs> inline BoolGeometry Geometry(Runs ctx, BooleanJob job) {
    return {ctx.template Run<packed_float3>(job.Positions), ctx.template Run<packed_float3>(job.Normals), ctx.template Run<packed_float3>(job.TriangleNormals), ctx.template Run<packed_uint3>(job.TriangleVertices), ctx.template Run<uint>(job.Pairs)};
}

template<typename Runs> inline uint BoolRecordHalfedge(Runs ctx, BooleanJob job, uint k) { return ctx.template Run<uint>(job.IntersectionRecords)[6u * k]; }
template<typename Runs> inline uint BoolRecordTriangle(Runs ctx, BooleanJob job, uint k) { return ctx.template Run<uint>(job.IntersectionRecords)[6u * k + 1u]; }
template<typename Runs> inline int BoolRecordCrossing(Runs ctx, BooleanJob job, uint k) { return as_type<int>(ctx.template Run<uint>(job.IntersectionRecords)[6u * k + 2u]); }
template<typename Runs> inline float3 BoolRecordPosition(Runs ctx, BooleanJob job, uint k) {
    device const uint *record = ctx.template Run<uint>(job.IntersectionRecords) + 6u * k;
    return float3(as_type<float>(record[3]), as_type<float>(record[4]), as_type<float>(record[5]));
}

template<typename Runs> inline uint BoolCount(Runs ctx, BooleanJob job, uint e) { return uint(abs(ctx.template Run<int>(job.Inclusion)[e])); }

// Scanned offsets into each run of the count scan.
template<typename Runs> inline uint BoolPolygonFirst(Runs ctx, BooleanJob job, uint t) { return ctx.template Run<uint>(job.Counts)[t]; }
template<typename Runs> inline uint BoolPartialFirst(Runs ctx, BooleanJob job, uint h) { return ctx.template Run<uint>(job.Counts)[BoolTriangleCount(job) + h] - job.PolygonHalfedges; }
template<typename Runs> inline uint BoolGroupFirst(Runs ctx, BooleanJob job, uint member) { return ctx.template Run<uint>(job.Counts)[4u * BoolTriangleCount(job) + member] - job.PolygonHalfedges - job.PartialEntries; }
template<typename Runs> inline uint BoolOutputFirst(Runs ctx, BooleanJob job, uint e) {
    return ctx.template Run<uint>(job.Counts)[4u * BoolTriangleCount(job) + 2u * job.Intersections + e] - job.PolygonHalfedges - job.PartialEntries - job.GroupEntries;
}
// The position of a vertex or intersection entry.
template<typename Runs> inline float3 BoolEntryPosition(Runs ctx, BooleanJob job, uint e) {
    const uint vertices = BoolVertexCount(job);
    return e < vertices ? float3(ctx.template Run<packed_float3>(job.Positions)[e]) : BoolRecordPosition(ctx, job, e - vertices);
}
template<typename Runs> inline float3 BoolOutputPosition(Runs ctx, BooleanJob job, uint d) { return BoolEntryPosition(ctx, job, ctx.template Run<uint>(job.OutputEntries)[d]); }

#endif
