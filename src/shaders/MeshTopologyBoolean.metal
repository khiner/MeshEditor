#ifndef MESHTOPOLOGYBOOLEAN_MSL
#define MESHTOPOLOGYBOOLEAN_MSL

// A boolean's output counts and emission from the runs its intersection passes computed.
// Count entries are its vertices, then its intersections, then its triangles, so the vertex scan matches the polygons' output vertices.
#include "MeshBooleanShared.metal"
#include "MeshTopologyContext.metal"

struct TopoBooleanRuns {
    device uint *Storage;
    template<typename T> device T *Run(uint offset) const { return reinterpret_cast<device T *>(Storage + offset); }
};
inline TopoBooleanRuns TopoBooleanStorage(TopoContext ctx, MeshTopologyJob job) { return {BindlessBufferMutable(uint, ctx.B.Buffer, job.Boolean.Slot)}; }

// The weights of p over a triangle's corners: exact near a corner, without the far corner near an edge, and by areas inside.
inline float3 TopoBarycentric(float3 p, float3 a, float3 b, float3 c) {
    const float3 corners[3] = {a, b, c};
    const float3 edges[3] = {c - b, a - c, b - a};
    const float3 d2 = float3(dot(edges[0], edges[0]), dot(edges[1], edges[1]), dot(edges[2], edges[2]));
    const uint longest = d2.x > d2.y && d2.x > d2.z ? 0u : d2.y > d2.z ? 1u : 2u;
    const float tolerance2 = d2[longest] * 1e-10f;
    for (uint i = 0u; i < 3u; ++i) {
        const float3 dv = p - corners[i];
        if (dot(dv, dv) <= tolerance2) return float3(i == 0u, i == 1u, i == 2u);
    }
    const float3 normal = cross(edges[0], edges[1]);
    const float area2 = dot(normal, normal);
    if (area2 > d2[longest] * tolerance2) {
        float3 uvw;
        for (uint i = 0u; i < 3u; ++i) {
            const float3 side = cross(edges[i], p - corners[(i + 1u) % 3u]);
            uvw[i] = dot(side, side) < d2[i] * tolerance2 ? 0.f : dot(side, normal);
        }
        const float sum = uvw.x + uvw.y + uvw.z;
        return sum == 0.f ? float3(1.f / 3.f) : uvw / sum;
    }
    const uint next = (longest + 1u) % 3u;
    const float alpha = d2[longest] > 0.f ? dot(p - corners[next], edges[longest]) / d2[longest] : 0.f;
    float3 uvw = float3(0.f);
    uvw[next] = 1.f - alpha;
    uvw[(next + 1u) % 3u] = alpha;
    return uvw;
}

// Each vertex and intersection entry counts its output copies, and each triangle entry its polygon's triangles.
kernel void TopologyBooleanCount(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint e = tile.y * ScanTileSize + lane;
    if (e >= job.CountEntries) return;
    const TopoBooleanRuns runs = TopoBooleanStorage(ctx, job);
    const BooleanJob boolean = *runs.Run<BooleanJob>(job.Boolean.Offset);
    const uint entries = BoolVertexCount(boolean) + boolean.Intersections, triangles = BoolTriangleCount(boolean);
    if (e < entries) ctx.WriteCounts(job, e, uint3(BoolCount(runs, boolean, e), 0u, 0u));
    else if (e < entries + triangles) {
        const uint count = runs.Run<uint>(boolean.PolygonTriangleCounts)[e - entries];
        ctx.WriteCounts(job, e, uint3(0u, count, 3u * count));
    } else ctx.WriteCounts(job, e, uint3(0u));
}

// A vertex copies its source vertex, displaced to its position in the first source's space.
// An intersection interpolates its halfedge's ends, displaced to the crossing.
// A triangle's polygon triangles take the source triangle's corners, weighted by each corner's position in it.
kernel void TopologyBooleanScatter(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint e = tile.y * ScanTileSize + lane;
    const TopoBooleanRuns runs = TopoBooleanStorage(ctx, job);
    const BooleanJob boolean = *runs.Run<BooleanJob>(job.Boolean.Offset);
    const uint vertices = BoolVertexCount(boolean), entries = vertices + boolean.Intersections;
    if (e >= entries + BoolTriangleCount(boolean)) return;
    const BoolGeometry g = Geometry(runs, boolean);
    if (e < entries) {
        const uint count = BoolCount(runs, boolean, e), first = ctx.Counts(job, TopoCountVertices)[e];
        const float3 p = BoolEntryPosition(runs, boolean, e);
        uint a = e, c = e;
        float weight = 0.f;
        if (e >= vertices) {
            const uint h = BoolRecordHalfedge(runs, boolean, e - vertices);
            a = g.Start(h);
            c = g.End(h);
            const float3 pa = g.Position(a), d = g.Position(c) - pa;
            const float length2 = dot(d, d);
            weight = length2 > 0.f ? clamp(dot(p - pa, d) / length2, 0.f, 1.f) : 0.f;
        }
        const float3 base = mix(ctx.SrcPosition(job, a), ctx.SrcPosition(job, c), weight);
        for (uint j = 0u; j < count; ++j) {
            ctx.WriteVertexMap(job, first + j, a, c, weight);
            *ctx.Inward(job, first + j, 0u) = packed_float3(p - base);
        }
        return;
    }
    const uint t = e - entries, triangles = runs.Run<uint>(boolean.PolygonTriangleCounts)[t];
    if (!triangles) return;
    uint fd = ctx.Counts(job, TopoCountFaces)[e], base = ctx.Counts(job, TopoCountCorners)[e];
    const uint polygon_first = BoolPolygonFirst(runs, boolean, t);
    device const packed_uint3 *polygon = runs.Run<packed_uint3>(boolean.Polygons) + polygon_first;
    device const uint *refs = runs.Run<uint>(boolean.PolygonTriangles) + 6u * polygon_first;
    device const uint *output_entries = runs.Run<uint>(boolean.OutputEntries);
    const uint3 source_vertices = uint3(runs.Run<packed_uint3>(boolean.TriangleVertices)[t]);
    const uint3 corners = uint3(runs.Run<packed_uint3>(boolean.TriangleCorners)[t]);
    const uint face = runs.Run<uint>(boolean.TriangleFaces)[t];
    const float3 corner_a = g.Position(source_vertices.x), corner_b = g.Position(source_vertices.y), corner_c = g.Position(source_vertices.z);
    for (uint i = 0u; i < triangles; ++i, ++fd, base += 3u) {
        for (uint k = 0u; k < 3u; ++k) {
            const uint ref = refs[3u * i + k], arriving = refs[3u * i + (k + 2u) % 3u];
            const uint output_vertex = uint3(polygon[ref & 0x7fffffffu]).x, entry = output_entries[output_vertex];
            // The side arriving at a corner keeps its source edge when it is a polygon halfedge along one.
            const uint source = (arriving & 0x80000000u) ? uint3(polygon[arriving & 0x7fffffffu]).z : InvalidOffset;
            const float3 w = entry == source_vertices.x ? float3(1.f, 0.f, 0.f) : entry == source_vertices.y ? float3(0.f, 1.f, 0.f) :
                entry == source_vertices.z ? float3(0.f, 0.f, 1.f) : TopoBarycentric(BoolEntryPosition(runs, boolean, entry), corner_a, corner_b, corner_c);
            ctx.WriteCorner4(job, base + k, output_vertex, uint4(corners.x, corners.y, corners.z, corners.z), w.x + w.y > 0.f ? w.y / (w.x + w.y) : 0.f, w.z, source, false);
        }
        TopoEmitFace(ctx, job, fd, base, 3u, face, false);
    }
}

#endif
