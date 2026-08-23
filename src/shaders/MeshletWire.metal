#include "MeshletResolve.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"
#include "WireMeshConstant.metal"

// Wireframe edges for the overlay pass.
// Each threadgroup transforms up to a line group of the draw's edge list into screen-space lines.
// Emission is unshared, two dedicated vertices per edge, so ElementStateColor can tint each halfedge.
using WireMeshOutput = metal::mesh<LineVaryings, void, OverlayDispatch_LineGroupLines * 2u, OverlayDispatch_LineGroupLines, metal::topology::line>;

// The object's selection tint outside edit mode, and under ElementStateColor the halfedge's own state.
inline float4 WireColor(const thread Scene &scene, DrawData draw, uint vertex_index) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float4 base = WireBaseColor(scene);
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        return scene.ObjectSelectionColor(scene.InstanceState(draw), base);
    }
    if (!ElementStateColor || draw.ElementStateSlotOffset.Slot == INVALID_SLOT) return base;

    const uint element_state = uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + vertex_index]);
    if ((element_state & STATE_ACTIVE) != 0u) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if ((element_state & STATE_SELECTED) != 0u) {
        return scene.View.InteractionMode == InteractionMode_Edit && scene.View.EditElement == Element_Edge ?
            float4(float3(colors.EdgeSelected), 1.0f) :
            float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    }
    return base;
}

[[mesh]] void MeshletWireMesh(
    WireMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_edge = threadgroup_position.x * OverlayDispatch_LineGroupLines;
    const uint edge_count = min(OverlayDispatch_LineGroupLines, pc.ElementCount - first_edge);
    output.set_primitive_count(edge_count);
    if (thread_index >= edge_count * 2u) return;

    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    const uint edge = first_edge + thread_index / 2u;
    // Position of this halfedge in the draw's line list, which is also its element-state index.
    const uint vertex_index = edge * 2u + (thread_index & 1u);
    const uint vertex_id = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_index];
    float4 clip = MeshletPosition(scene, draw, MeshletWorld(scene, draw), vertex_id);
    clip.z -= NdcOffsetFactor(scene); // Push lines in front of faces.
    output.set_vertex(thread_index, MakeLineVertex(clip, WireColor(scene, draw, vertex_index), float2(scene.View.ViewportSize)));
    output.set_index(thread_index, thread_index);
}
