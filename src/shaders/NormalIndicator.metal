#ifndef NORMALINDICATOR_MSL
#define NORMALINDICATOR_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/MeshletLimit.h"
#include "MeshletResolve.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "LineQuad.metal"
#include "VertexBlocks.metal"

// Emits normal-indicator line groups scaled to local geometry size.
constant float NormalIndicatorLengthScale = 0.25f;
constant uint NormalIndicatorThreads = uint(MeshletLimit::MaxVertices);
constant uint NormalIndicatorSimdGroups = NormalIndicatorThreads / 32u;
using NormalIndicatorOutput = metal::mesh<EdgeQuadVaryings, void, NormalIndicatorThreads * 4u, NormalIndicatorThreads * 2u, metal::topology::triangle>;

inline float MeanIncidentEdgeLength(const thread Scene &scene, DrawData draw, uint vertex_id, float3 position) {
    const ConnectivityView conn{scene.B, draw.Connectivity, draw.FaceCount};
    float total = 0.0f;
    uint count = 0u;
    const auto add = [&](uint corner) {
        total += length(scene.GetLocalPosition(draw, scene.CornerVertexOrdinal(draw, corner)) - position);
        ++count;
    };
    // Each fan corner ends at the vertex, so its incoming edge starts at its opposite's corner, or at its previous corner on a boundary.
    // A boundary edge leaving the vertex ends at the next corner.
    for (const auto item : conn.Fan(draw.VertexOffset + vertex_id)) {
        const uint h = item.x;
        if (conn.IncomingEdge(h) != InvalidOffset) {
            const uint opposite = conn.Opposite(h);
            add(opposite != InvalidOffset ? opposite : conn.Previous(h));
        }
        if (conn.BoundaryOutgoingEdge(h) != InvalidOffset) add(conn.Next(h));
    }
    return count ? total / float(count) : 0.0f;
}

// Returns a vertex's local-space indicator segment scaled to its incident edges.
inline void VertexNormalSegment(const thread Scene &scene, DrawData draw, uint vertex_id, thread float3 &start, thread float3 &end) {
    start = scene.GetLocalPosition(draw, vertex_id);
    const float3 normal = scene.GetVertexNormal(draw, vertex_id);
    end = start + NormalIndicatorLengthScale * MeanIncidentEdgeLength(scene, draw, vertex_id, start) * normal;
}

// Returns a face's local-space indicator segment scaled to its area.
inline void FaceNormalSegment(const thread Scene &scene, DrawData draw, uint element, thread float3 &start, thread float3 &end) {
    const ConnectivityView conn{scene.B, draw.Connectivity, draw.FaceCount};
    const uint2 loop = conn.FaceHalfedges(element);
    float3 sum = float3(0);
    for (uint h = loop.x; h < loop.y; ++h) sum += scene.GetLocalPosition(draw, scene.CornerVertexOrdinal(draw, h));
    float area = 0.f;
    const uint first = scene.FaceTriangles(scene.View.FaceTriangleStartSlot)[element];
    for (uint i = 0u; i < loop.y - loop.x - 2u; ++i) {
        const uint3 h = uint3(BindlessBuffer(packed_uint3, scene.B.Buffer, draw.TriangleSlot)[first + i]);
        const float3 a = scene.GetLocalPosition(draw, scene.CornerVertexOrdinal(draw, h.x));
        const float3 b = scene.GetLocalPosition(draw, scene.CornerVertexOrdinal(draw, h.y));
        const float3 c = scene.GetLocalPosition(draw, scene.CornerVertexOrdinal(draw, h.z));
        area += 0.5f * length(cross(b - a, c - a));
    }
    start = sum / float(loop.y - loop.x);
    end = start + NormalIndicatorLengthScale * sqrt(area) * scene.GetFaceNormal(draw, element);
}

// Emits one stroke per present lane from its local-space segment.
inline void EmitNormalIndicator(
    thread NormalIndicatorOutput output, uint thread_index, uint lane, threadgroup uint *simd_counts,
    const thread Scene &scene, DrawData draw, uint element, bool faces
) {
    const uint present = element != InvalidOffset ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, NormalIndicatorSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (present == 0u) return;

    const Transform world = MeshletWorld(scene, draw);
    float3 start, end;
    if (faces) FaceNormalSegment(scene, draw, element, start, end);
    else VertexNormalSegment(scene, draw, element, start, end);

    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float4 color = float4(float3(faces ? colors.FaceNormal : colors.VertexNormal), 1.0f);
    float4 clip[2];
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, endpoint == 0u ? start : end));
        clip[endpoint] = scene.ViewProj() * float4(world_pos, 1.0f);
        clip[endpoint].z -= NdcOffsetFactor(scene);
    }
    EmitStroke(output, compact.x, scene, clip[0], clip[1], color);
}

// Each finest cluster emits the faces whose first triangle it holds.
[[mesh]] void FaceNormalIndicatorMesh(
    NormalIndicatorOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[NormalIndicatorSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(scene, pc, threadgroup_position.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    uint element = InvalidOffset;
    if (!MeshletCoarse(work.Meshlet) && thread_index < work.Meshlet.TriangleCount) {
        const uint triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[work.Meshlet.TriangleOffset + thread_index];
        const uint face = scene.TriangleFace(work.Draw, triangle);
        if (face != InvalidOffset && triangle == scene.FaceTriangles(scene.View.FaceTriangleStartSlot)[face]) element = face;
    }
    EmitNormalIndicator(output, thread_index, lane, simd_counts, scene, work.Draw, element, true);
}

// Each lane of a canonical vertex block emits its live vertex.
[[mesh]] void VertexNormalIndicatorMesh(
    NormalIndicatorOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VertexBlockPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[NormalIndicatorSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    // A stroke spans its half width around the vertex, pulled toward the camera as EmitNormalIndicator pulls it.
    const VertexBlockLane work = ResolveVertexBlockLane(
        scene, pc, threadgroup_position.x, thread_index, scene.Theme.EdgeWidth + 0.5f, NdcOffsetFactor(scene)
    );
    EmitNormalIndicator(output, thread_index, lane, simd_counts, scene, work.Draw, work.VertexId, false);
}

#endif
