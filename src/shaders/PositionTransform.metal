#ifndef POSITIONTRANSFORM_MSL
#define POSITIONTRANSFORM_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"

// Line and point pick ids for the selection pass, emitted from the mesh's own index buffers.
// Each threadgroup covers a line group of edges or a point group of one instance.
constant uint SelectionLineEdges = OverlayDispatch_LineGroupLines;
constant uint SelectionPoints = OverlayDispatch_PointGroupPoints;
using SelectionLineIdOutput = metal::mesh<ObjectIdFragmentVaryings, void, SelectionLineEdges * 2u, SelectionLineEdges, metal::topology::line>;
using SelectionPointIdOutput = metal::mesh<ObjectIdVaryings, void, SelectionPoints, SelectionPoints, metal::topology::point>;

// Clip position of the draw's vertex at `index_position` in its index buffer.
inline float4 SelectionIdClip(const thread Scene &scene, DrawData draw, uint index_position) {
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + index_position];
    return MeshletPosition(scene, draw, scene.Models(draw.ModelSlot)[draw.FirstInstance], idx);
}

inline uint SelectionObjectId(const thread Scene &scene, DrawData draw) {
    return draw.ObjectIdSlot != INVALID_SLOT ? scene.ObjectIds(draw.ObjectIdSlot)[draw.FirstInstance] : 0u;
}

[[mesh]] void SelectionLineIdMesh(
    SelectionLineIdOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    if (!InstanceInFrustum(scene, draw)) {
        output.set_primitive_count(0u);
        return;
    }
    const uint first_edge = threadgroup_position.x * SelectionLineEdges;
    const uint edge_count = min(SelectionLineEdges, pc.ElementCount - first_edge);
    output.set_primitive_count(edge_count);
    if (thread_index >= edge_count * 2u) return;

    ObjectIdFragmentVaryings out;
    out.ObjectId = SelectionObjectId(scene, draw);
    out.Position = SelectionIdClip(scene, draw, first_edge * 2u + thread_index);
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

[[mesh]] void SelectionPointIdMesh(
    SelectionPointIdOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    if (!InstanceInFrustum(scene, draw)) {
        output.set_primitive_count(0u);
        return;
    }
    const uint first_point = threadgroup_position.x * SelectionPoints;
    const uint point_count = min(SelectionPoints, pc.ElementCount - first_point);
    output.set_primitive_count(point_count);
    if (thread_index >= point_count) return;

    ObjectIdVaryings out;
    out.ObjectId = SelectionObjectId(scene, draw);
    out.Position = SelectionIdClip(scene, draw, first_point + thread_index);
    out.PointSize = PointSize;
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

#endif
