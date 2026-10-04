#ifndef MESHLET_EDIT_OVERLAY_MSL
#define MESHLET_EDIT_OVERLAY_MSL

#include "ElementOverlay.metal"
#include "EditSelection.metal"
#include "MeshletEditGeometry.metal"
#include "gpu/MeshletLimit.h"
#include "MeshletResolve.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

constant uint MeshletEditSimdGroups = 5u;
using MeshletEditEdgeOutput = metal::mesh<EdgeQuadVaryings, void, uint(MeshletLimit::MaxTriangles) * 4u, uint(MeshletLimit::MaxTriangles) * 2u, metal::topology::triangle>;
using MeshletSelectEdgeOutput = metal::mesh<ElementIdFragmentVaryings, void, uint(MeshletLimit::MaxTriangles) * 2u, uint(MeshletLimit::MaxTriangles), metal::topology::line>;
using MeshletSelectEdgePointOutput = metal::mesh<ElementIdVaryings, void, uint(MeshletLimit::MaxTriangles) * 2u, uint(MeshletLimit::MaxTriangles) * 2u, metal::topology::point>;
using MeshletSelectFacePointOutput = metal::mesh<ElementIdVaryings, void, uint(MeshletLimit::MaxTriangles) * 3u, uint(MeshletLimit::MaxTriangles) * 3u, metal::topology::point>;

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
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(scene, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    MeshletEditEdgeGeometry geometry;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, geometry);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (!present) return;

    const uint vertex_base = compact.x * 4u;
    const bool edit_edge = scene.View.EditElement == Element::Edge;
    const auto color = [&](uint vertex_id) {
        return EditEdgeColor(scene, EditEdgeEndpointState(scene, work.Draw, geometry.Edge, vertex_id), edit_edge);
    };
    EditEdgeOverlay edge{
        geometry.Clip0, geometry.Clip1, color(geometry.Vertex0), color(geometry.Vertex1),
        pc.EdgeSharpnessSlot != InvalidSlot &&
            uint(scene.Bytes(pc.EdgeSharpnessSlot)[work.Draw.EditEdgeSharpnessOffset + geometry.Edge]) != 0u,
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
    const MeshletWork work = ResolveMeshletWork(scene, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    MeshletEditEdgeGeometry edge;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, edge);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y);
    if (!present) return;
    ElementIdFragmentVaryings out{.Position = edge.Clip0, .ElementId = work.Draw.ElementIdOffset + edge.Edge + 1u};
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
    const MeshletWork work = ResolveMeshletWork(scene, pc, group.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    MeshletEditEdgeGeometry edge;
    const auto present = ResolveMeshletEditEdgeCandidate(scene, work, bindless, pc, thread_index, pc.EditEdgeCorner, edge);
    const uint2 compact = CompactPresent(present, thread_index, lane, simd_counts, MeshletEditSimdGroups);
    if (thread_index == 0u) output.set_primitive_count(compact.y * 2u);
    if (!present) return;
    ElementIdVaryings out{.Position = edge.Clip0, .PointSize = 2.0f, .ElementId = work.Draw.ElementIdOffset + edge.Edge + 1u};
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
    const MeshletWork work = ResolveMeshletWork(scene, pc, group.x);
    if (!work.Valid || MeshletPrimitiveTopology(work.Meshlet) != uint(MeshPrimitiveTopology::Triangle) || MeshletCoarse(work.Meshlet)) {
        output.set_primitive_count(0u);
        return;
    }
    if (thread_index == 0u) output.set_primitive_count(work.Meshlet.TriangleCount * 3u);
    if (thread_index >= work.Meshlet.TriangleCount) return;
    const uint source_triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[work.Meshlet.TriangleOffset + thread_index];
    const MeshletTriangleCorners corners = ResolveMeshletCorners(
        scene, work.Draw, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot,
        work.Meshlet, source_triangle, thread_index
    );
    const uint face_id = scene.TriangleFace(work.Draw, source_triangle);
    const Transform world = MeshletWorld(scene, work.Draw);
    for (uint corner = 0u; corner < 3u; ++corner) {
        ElementIdVaryings out{.Position = MeshletPosition(scene, work.Draw, world, corners.VertexIds[corner]), .PointSize = 1.0f, .ElementId = scene.FacePickId(work.Draw, face_id)};
        const uint output_index = thread_index * 3u + corner;
        output.set_vertex(output_index, out);
        output.set_index(output_index, output_index);
    }
}

#endif
