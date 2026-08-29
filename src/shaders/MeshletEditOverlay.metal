#ifndef MESHLET_EDIT_OVERLAY_MSL
#define MESHLET_EDIT_OVERLAY_MSL

#include "ElementOverlay.metal"
#include "EditSelection.metal"
#include "MeshletEditGeometry.metal"
#include "MeshletLimit.metal"
#include "MeshletInstanceFlag.metal"
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

inline uint MeshletOwnedVertex(
    const thread Scene &scene, const thread MeshletWork &work,
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint local_vertex
) {
    if (local_vertex >= work.Meshlet.VertexCount) return INVALID_OFFSET;
    const uint packed = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, work.Meshlet, local_vertex);
    if ((packed & MeshletGeometryEncoding_EditVertexOwnerBit) == 0u) return INVALID_OFFSET;
    return MeshletVertexId(scene, work.Draw, MeshletPrimitiveTopology(work.Meshlet), packed);
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
    MeshletEditEdgeGeometry geometry;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, edge_corner, geometry);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (!present) return;

    const uint vertex_base = compact.x * 4u;
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

    const uint vertex_id = MeshletOwnedVertex(scene, work, bindless, pc, thread_index);
    const bool sound_point = (pc.RequiredInstanceFlags & MeshletInstanceFlag_SoundPoint) != 0u;
    const uint vertex_state = vertex_id != INVALID_OFFSET ? EditVertexState(scene, work.Draw, vertex_id) : 0u;
    const auto present = vertex_id != INVALID_OFFSET && (!sound_point || (vertex_state & STATE_SELECTED) != 0u);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditPointSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;

    PointVaryings out = ElementPointSprite(
        scene, work.Draw,
        MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id), vertex_id
    );
    if (sound_point) {
        constant ViewportThemeColors &colors = scene.Theme.Colors;
        out.Color = vertex_id == work.Instance.ExcitedVertex ? float4(colors.ElementExcited) :
            vertex_id == work.Instance.ActiveVertex ? float4(float4(colors.ElementActive).rgb, 1.0f) :
                                                     float4(float3(colors.VertexSelected), 1.0f);
    }
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
    const uint vertex_id = MeshletOwnedVertex(scene, work, bindless, pc, thread_index);
    const bool sound_point = (pc.RequiredInstanceFlags & MeshletInstanceFlag_SoundPoint) != 0u;
    const auto present = vertex_id != INVALID_OFFSET && (!sound_point || EditSelectionBit(scene, work.Draw.Selection.VertexBits, vertex_id));
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditPointSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;

    ElementIdVaryings out;
    out.ElementId = (sound_point ? 0u : work.Draw.ElementIdOffset) + vertex_id + 1u;
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
    MeshletEditEdgeGeometry edge;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, edge);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;
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
    MeshletEditEdgeGeometry edge;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, edge);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (!present) return;
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
    if (!work.Valid || MeshletPrimitiveTopology(work.Meshlet) != MeshPrimitiveTopology_Triangle || MeshletCoarse(work.Meshlet)) {
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
