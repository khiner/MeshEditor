#ifndef MESHTOPOLOGYWIREFRAME_MSL
#define MESHTOPOLOGYWIREFRAME_MSL

#include "MeshTopologyContext.metal"

// Geometry follows Blender's bmesh_wireframe.cc: two normal-offset copies per
// vertex, one inset per selected corner, and an outward inset at open boundaries.
// Each source edge owns a selected-face count and its selected corner. Counting
// all users also handles nonmanifold edges; an opposite alone cannot do that.
inline device uint *TopoWireEdges(TopoContext ctx, MeshTopologyJob job) {
    return ctx.Scratch() + job.HalfedgeAuxOffset;
}
inline bool TopoWireBoundary(TopoContext ctx, MeshTopologyJob job, uint e) {
    return e < job.SrcEdgeCount && TopoWireEdges(ctx, job)[2u * e] == 1u;
}
inline uint TopoWireCopy(TopoContext ctx, MeshTopologyJob job, uint v) {
    return ctx.Counts(job, TopoCountVertices)[v] + uint(TopoVertexKept(ctx, job, v));
}
inline uint TopoWireBoundaryEdges(TopoContext ctx, MeshTopologyJob job, uint v, thread uint2 &edges) {
    uint count = 0u;
    ctx.Src(job).ForEachIncidentEdge(ctx.SrcVertexDomain(job).Handle(v), [&](uint handle) {
        const uint e = ctx.SrcEdgeDomain(job).Index(handle);
        if (TopoWireBoundary(ctx, job, e) && count < 2u) edges[count++] = e;
    });
    return count;
}

// Blender's relative thickness averages lengths to tagged neighbors over the
// vertex's entire edge degree. Compute it once per affected vertex.
inline float TopoWireRelative(TopoContext ctx, MeshTopologyJob job, uint v) {
    if (!(job.Flags & TopologyFlagWireRelative)) return 1.f;
    float length = 0.f;
    uint count = 0u;
    const auto corners = ctx.SrcCorners(job);
    ctx.Src(job).ForEachIncidentEdge(ctx.SrcVertexDomain(job).Handle(v), [&](uint edge) {
        const uint h = ctx.Src(job).EdgeHalfedge(edge);
        const uint a = corners[h], b = corners[ctx.SrcPrev(job, h)];
        const uint other = a == v ? b : a;
        if (other < job.SrcVertexCount && (ctx.FlagVertices(job)[other] & TopoInRegion))
            length += distance(ctx.SrcPosition(job, v), ctx.SrcPosition(job, other));
        ++count;
    });
    return count ? length / float(count) : 0.f;
}
inline float TopoWireShell(float3 a, float3 b) {
    return rsqrt(max(0.5f * (1.f - clamp(dot(a, b), -1.f, 1.f)), 1e-8f));
}

inline void TopoWireVertices(TopoContext ctx, MeshTopologyJob job, uint v, uint first) {
    const float fac = TopoWireRelative(ctx, job, v);
    const float radius = job.Param0 * 0.5f * fac, mid = radius * job.Param1;
    const float3 normal = NormalizeOrZero(float3(ctx.SrcVertexNormals(job)[v]));
    *ctx.Inward(job, first, 0u) = packed_float3(normal * (mid - radius));
    *ctx.Inward(job, first + 1u, 0u) = packed_float3(normal * (mid + radius));
    // The face pass reuses the relative factor without revisiting this fan.
    *ctx.Inward(job, first, 1u) = packed_float3(fac, 0.f, 0.f);
    if (!(ctx.FlagVertices(job)[v] & TopoOnBoundary)) return;
    uint2 edges;
    const uint count = TopoWireBoundaryEdges(ctx, job, v, edges);
    const auto corners = ctx.SrcCorners(job);
    float3 directions[2], normals[2], inward = float3(0.f);
    const float3 p = ctx.SrcPosition(job, v);
    for (uint i = 0u; i < count; ++i) {
        const uint h = TopoWireEdges(ctx, job)[2u * edges[i] + 1u];
        const uint a = corners[ctx.SrcPrev(job, h)], b = corners[h];
        directions[i] = NormalizeOrZero(ctx.SrcPosition(job, a == v ? b : a) - p);
        normals[i] = NormalizeOrZero(float3(ctx.SrcFaceNormals(job)[ctx.SrcFaceOf(job, h)]));
        inward += NormalizeOrZero(cross(normals[i], ctx.SrcPosition(job, b) - ctx.SrcPosition(job, a)));
    }
    const float3 face_normal = count == 2u ? NormalizeOrZero(normals[0] + normals[1]) : normals[0];
    const float3 edge_direction = count == 2u ? directions[1] - directions[0] : -directions[0];
    float3 tangent = NormalizeOrZero(cross(edge_direction, face_normal));
    if (dot(tangent, inward) > 0.f) tangent = -tangent;
    float shell = 1.f;
    if (count == 2u && (job.Flags & TopologyFlagEvenOffset)) {
        const float3 a = NormalizeOrZero(directions[0] - face_normal * dot(face_normal, directions[0]));
        const float3 b = NormalizeOrZero(directions[1] - face_normal * dot(face_normal, directions[1]));
        shell = TopoWireShell(a, b);
    }
    *ctx.Inward(job, first + 2u, 0u) = packed_float3(tangent * (radius * shell) + normal * mid);
}

inline uint3 TopoWireFaceCounts(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    if (!ctx.SrcSelectedFace(job, f)) return uint3(0u, 1u, n);
    uint sides = 2u * n;
    if (job.Flags & TopologyFlagWireBoundary)
        for (uint h = range.x; h < range.y; ++h) sides += 2u * uint(TopoWireBoundary(ctx, job, ctx.SrcEdge(job, h)));
    const bool keep = !(job.Flags & TopologyFlagWireReplace);
    return uint3(n, sides + uint(keep), 4u * sides + (keep ? n : 0u));
}

inline void TopoWireFace(TopoContext ctx, MeshTopologyJob job, uint f, uint fd, uint base, uint vertices) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const auto corners = ctx.SrcCorners(job);
    if (!(job.Flags & TopologyFlagWireReplace)) {
        for (uint h = range.x; h < range.y; ++h)
            ctx.WriteCorner(job, base + h - range.x, ctx.Counts(job, TopoCountVertices)[corners[h]], h, h, 0.f, h, false);
        TopoEmitFace(ctx, job, fd++, base, range.y - range.x, f, false);
        base += range.y - range.x;
    }
    const float3 face_normal = NormalizeOrZero(float3(ctx.SrcFaceNormals(job)[f]));
    for (uint h = range.x; h < range.y; ++h) {
        const uint v = corners[h], d = vertices + h - range.x;
        const float3 p = ctx.SrcPosition(job, v);
        const float3 a = NormalizeOrZero(ctx.SrcPosition(job, corners[ctx.SrcPrev(job, h)]) - p);
        const float3 b = NormalizeOrZero(ctx.SrcPosition(job, corners[ctx.SrcNext(job, h)]) - p);
        float3 local_normal = cross(a, -b);
        if (dot(local_normal, face_normal) < 0.f) local_normal = -local_normal;
        if (dot(local_normal, local_normal) < 1e-12f) local_normal = face_normal;
        const float3 tangent = NormalizeOrZero(cross(a - b, local_normal));
        const float radius = 0.5f * job.Param0 * float3(*ctx.Inward(job, TopoWireCopy(ctx, job, v), 1u)).x;
        const float shell = job.Flags & TopologyFlagEvenOffset ? TopoWireShell(a, b) : 1.f;
        ctx.WriteVertexMap(job, d, v, v, 0.f);
        ctx.SelectDstVertex(job, d);
        *ctx.Inward(job, d, 0u) = packed_float3(tangent * (radius * shell) + NormalizeOrZero(float3(ctx.SrcVertexNormals(job)[v])) * (radius * job.Param1));
    }
    const auto quad = [&](uint4 loop, uint4 sources) {
        for (uint k = 0u; k < 4u; ++k)
            ctx.WriteCorner(job, base + k, loop[k], sources[k], sources[k], 0.f, InvalidOffset, true);
        TopoEmitFace(ctx, job, fd++, base, 4u, f, true);
        base += 4u;
    };
    for (uint h = range.x; h < range.y; ++h) {
        const uint next = ctx.SrcNext(job, h), a = TopoWireCopy(ctx, job, corners[h]), b = TopoWireCopy(ctx, job, corners[next]);
        const uint ca = vertices + h - range.x, cb = vertices + next - range.x;
        quad(uint4(ca, cb, b, a), uint4(h, next, next, h));
        quad(uint4(cb, ca, a + 1u, b + 1u), uint4(next, h, h, next));
        if ((job.Flags & TopologyFlagWireBoundary) && TopoWireBoundary(ctx, job, ctx.SrcEdge(job, next))) {
            quad(uint4(b + 2u, a + 2u, a, b), uint4(next, h, h, next));
            quad(uint4(a + 2u, b + 2u, b + 1u, a + 1u), uint4(h, next, next, h));
        }
    }
}
#endif
