#ifndef MESHLET_EDIT_OVERLAY_MSL
#define MESHLET_EDIT_OVERLAY_MSL

#include "ElementOverlay.metal"
#include "EditSelection.metal"
#include "MeshletEditEdgeEncoding.metal"
#include "MeshletLimit.metal"
#include "MeshletResolve.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

constant uint MeshletEditSimdGroups = 5u;
constant uint MeshletEditPointSimdGroups = 2u;
using MeshletEditEdgeOutput = metal::mesh<EdgeQuadVaryings, void, MeshletLimit_MaxTriangles * 4u, MeshletLimit_MaxTriangles * 2u, metal::topology::triangle>;
using MeshletEditPointOutput = metal::mesh<PointVaryings, void, MeshletLimit_MaxVertices, MeshletLimit_MaxVertices, metal::topology::point>;
using MeshletSelectPointOutput = metal::mesh<ElementIdVaryings, void, MeshletLimit_MaxVertices, MeshletLimit_MaxVertices, metal::topology::point>;
using MeshletSelectEdgeOutput = metal::mesh<ElementIdFragmentVaryings, void, MeshletLimit_MaxTriangles * 2u, MeshletLimit_MaxTriangles, metal::topology::line>;
using MeshletSelectEdgePointOutput = metal::mesh<ElementIdVaryings, void, MeshletLimit_MaxTriangles * 2u, MeshletLimit_MaxTriangles * 2u, metal::topology::point>;
using MeshletSelectFacePointOutput = metal::mesh<ElementIdVaryings, void, MeshletLimit_MaxTriangles * 3u, MeshletLimit_MaxTriangles * 3u, metal::topology::point>;

inline uint MeshletPackedEditEdge(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    const thread MeshletWork &work, uint local_triangle, uint edge_corner
) {
    const uint source_triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
        work.Meshlet.TriangleOffset + local_triangle
    ];
    return BindlessBuffer(uint, bindless.Buffer, pc.MeshletEditEdgeSlot)[
        work.Draw.EditEdgeOffset + source_triangle * 3u + edge_corner
    ];
}

inline uint2 CompactPresent(
    uint present, uint thread_index, uint lane, threadgroup uint *simd_counts, uint simd_group_count
) {
    const uint simd_rank = simd_prefix_exclusive_sum(present);
    const uint simd_count = simd_sum(present);
    const uint simdgroup = thread_index / 32u;
    if (lane == 0u) simd_counts[simdgroup] = simd_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = simd_rank;
    for (uint i = 0u; i < simdgroup; ++i) rank += simd_counts[i];
    uint count = 0u;
    for (uint i = 0u; i < simd_group_count; ++i) count += simd_counts[i];
    return uint2(rank, count);
}

struct MeshletEditEdgeGeometry {
    float4 Clip0, Clip1;
    uint Edge, Vertex0, Vertex1;
};

inline MeshletEditEdgeGeometry ResolveMeshletEditEdge(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc,
    uint local_triangle, uint edge_corner, uint packed_edge
) {
    device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot);
    const uint triangle_base = MeshletLocalTriangleOffset(work.Meshlet) + local_triangle * 3u;
    const uint local0 = uint(triangles[triangle_base + edge_corner] & MeshletGeometryEncoding_LocalIndexMask);
    const uint local1 = uint(triangles[triangle_base + (edge_corner + 1u) % 3u] & MeshletGeometryEncoding_LocalIndexMask);
    const uint packed0 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local0);
    const uint packed1 = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local1);
    const uint vertex0 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed0);
    const uint vertex1 = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed1);
    const Transform world = MeshletWorld(scene, work.Draw);
    return {
        MeshletPosition(scene, work.Draw, world, vertex0),
        MeshletPosition(scene, work.Draw, world, vertex1),
        packed_edge & MeshletEditEdgeEncoding_EdgeMask,
        vertex0,
        vertex1,
    };
}

inline void EmitMeshletEditEdge(
    thread MeshletEditEdgeOutput output, uint thread_index, uint lane, uint group_index,
    threadgroup uint *simd_counts,
    device const BindlessSet &bindless, constant SceneViewUBO &view,
    constant ViewportTheme &theme, constant WorkspaceLights &workspace,
    constant MeshletDrawPushConstants &pc, uint edge_corner
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group_index);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    uint packed_edge = INVALID_OFFSET;
    if (thread_index < work.Meshlet.TriangleCount) {
        packed_edge = MeshletPackedEditEdge(bindless, pc, work, thread_index, edge_corner);
    }
    const uint present = packed_edge != INVALID_OFFSET ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (present == 0u) return;

    const uint vertex_base = compact.x * 4u;
    const MeshletEditEdgeGeometry geometry = ResolveMeshletEditEdge(
        scene, work, bindless, pc, thread_index, edge_corner, packed_edge
    );
    const bool edit_edge = scene.View.EditElement == Element_Edge;
    EditEdgeOverlay edge{
        geometry.Clip0, geometry.Clip1,
        EditEdgeColor(scene, EditEdgeEndpointState(scene, work.Draw, geometry.Edge, geometry.Vertex0), edit_edge),
        EditEdgeColor(scene, EditEdgeEndpointState(scene, work.Draw, geometry.Edge, geometry.Vertex1), edit_edge),
        pc.EdgeSharpnessSlot != INVALID_SLOT &&
            uint(scene.Bytes(pc.EdgeSharpnessSlot)[work.Instance.EditEdgeSharpnessOffset + geometry.Edge]) != 0u,
    };
    edge.Clip0.z -= NdcOffsetFactor(scene);
    edge.Clip1.z -= NdcOffsetFactor(scene);
    for (uint corner = 0u; corner < 4u; ++corner) {
        output.set_vertex(vertex_base + corner, EditEdgeQuadCorner(scene, edge, corner));
    }
    for (uint i = 0u; i < 6u; ++i) {
        output.set_index(compact.x * 6u + i, vertex_base + LineQuadCornerLut[i]);
    }
}

[[mesh]] void MeshletEditEdgeMesh(
    MeshletEditEdgeOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditSimdGroups];
    EmitMeshletEditEdge(
        output, thread_index, lane, group.x, simd_counts, bindless, view, theme, workspace, pc, pc.EditEdgeCorner
    );
}

[[mesh]] void MeshletEditPointMesh(
    MeshletEditPointOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditPointSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }

    const uint packed = thread_index < work.Meshlet.VertexCount ?
        MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index) : 0u;
    const bool local_owner = thread_index < work.Meshlet.VertexCount &&
        (packed & MeshletGeometryEncoding_EditVertexOwnerBit) != 0u;
    const uint vertex_id = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed);
    const uint present = local_owner ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditPointSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (present == 0u) return;

    const uint state = EditVertexState(scene, work.Draw, vertex_id);
    const PointVaryings out = EditPointSprite(
        scene, MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id),
        EditVertexColor(scene, state)
    );
    output.set_vertex(compact.x, out);
    output.set_index(compact.x, compact.x);
}

[[mesh]] void MeshletSelectPointMesh(
    MeshletSelectPointOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditPointSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint packed = thread_index < work.Meshlet.VertexCount ?
        MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, thread_index) : 0u;
    const uint present = thread_index < work.Meshlet.VertexCount &&
        (packed & MeshletGeometryEncoding_EditVertexOwnerBit) != 0u ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditPointSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (present == 0u) return;

    const uint vertex_id = MeshletVertexId(scene, work.Draw, MeshPrimitiveTopology_Triangle, packed);
    ElementIdVaryings out;
    out.ElementId = work.Draw.ElementIdOffset + vertex_id + 1u;
    out.Position = MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id);
    out.PointSize = PointSize;
    output.set_vertex(compact.x, out);
    output.set_index(compact.x, compact.x);
}

[[mesh]] void MeshletSelectEdgeMesh(
    MeshletSelectEdgeOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint packed_edge = thread_index < work.Meshlet.TriangleCount ?
        MeshletPackedEditEdge(bindless, pc, work, thread_index, pc.EditEdgeCorner) : INVALID_OFFSET;
    const uint present = packed_edge != INVALID_OFFSET ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (present == 0u) return;
    const MeshletEditEdgeGeometry edge = ResolveMeshletEditEdge(
        scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, packed_edge
    );
    ElementIdFragmentVaryings out;
    out.ElementId = work.Draw.ElementIdOffset + edge.Edge + 1u;
    out.Position = edge.Clip0;
    output.set_vertex(compact.x * 2u, out);
    out.Position = edge.Clip1;
    output.set_vertex(compact.x * 2u + 1u, out);
    output.set_index(compact.x * 2u, compact.x * 2u);
    output.set_index(compact.x * 2u + 1u, compact.x * 2u + 1u);
}

[[mesh]] void MeshletSelectEdgePointMesh(
    MeshletSelectEdgePointOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint simd_counts[MeshletEditSimdGroups];
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint packed_edge = thread_index < work.Meshlet.TriangleCount ?
        MeshletPackedEditEdge(bindless, pc, work, thread_index, pc.EditEdgeCorner) : INVALID_OFFSET;
    const uint present = packed_edge != INVALID_OFFSET ? 1u : 0u;
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (present == 0u) return;
    const MeshletEditEdgeGeometry edge = ResolveMeshletEditEdge(
        scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, packed_edge
    );
    ElementIdVaryings out;
    out.ElementId = work.Draw.ElementIdOffset + edge.Edge + 1u;
    out.PointSize = 2.0f;
    out.Position = edge.Clip0;
    output.set_vertex(compact.x * 2u, out);
    output.set_index(compact.x * 2u, compact.x * 2u);
    out.Position = edge.Clip1;
    output.set_vertex(compact.x * 2u + 1u, out);
    output.set_index(compact.x * 2u + 1u, compact.x * 2u + 1u);
}

[[mesh]] void MeshletSelectFacePointMesh(
    MeshletSelectFacePointOutput output, uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, group.x);
    if (!work.Valid || MeshletCoarse(work.Meshlet)) {
        output.set_primitive_count(0u);
        return;
    }
    if (thread_index == 0u) output.set_primitive_count(work.Meshlet.TriangleCount * 3u);
    if (thread_index >= work.Meshlet.TriangleCount) return;
    const uint source_triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[work.Meshlet.TriangleOffset + thread_index];
    const MeshletTriangleCorners corners = ResolveMeshletCorners(
        scene, work.Draw, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot,
        work.Meshlet, work.Primitive, source_triangle, thread_index
    );
    const uint face_id = scene.ObjectIds(work.Draw.ObjectIdSlot)[
        work.Draw.FaceIdOffset + source_triangle - work.Primitive.FirstTriangle
    ];
    const Transform world = MeshletWorld(scene, work.Draw);
    for (uint corner = 0u; corner < 3u; ++corner) {
        ElementIdVaryings out;
        out.ElementId = work.Draw.ElementIdOffset + face_id;
        out.Position = MeshletPosition(scene, work.Draw, world, corners.VertexIds[corner]);
        out.PointSize = 1.0f;
        const uint output_index = thread_index * 3u + corner;
        output.set_vertex(output_index, out);
        output.set_index(output_index, output_index);
    }
}

#endif
