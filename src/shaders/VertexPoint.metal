#ifndef VERTEXPOINT_MSL
#define VERTEXPOINT_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "MainDrawPushConstants.metal"

// Object mode has no element selection, so points take their object's selection color.
inline float4 point_color(const thread Scene &scene, DrawData draw, uint element_state) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        return scene.ObjectSelectionColor(scene.InstanceState(draw), float4(float3(colors.Vertex), 1.0f));
    }
    if ((element_state & STATE_EXCITED) != 0u) return float4(colors.ElementExcited);
    if ((element_state & STATE_ACTIVE) != 0u) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if ((element_state & STATE_SELECTED) != 0u) return float4(float3(colors.VertexSelected), 1.0f);
    return float4(float3(colors.Vertex), 1.0f);
}

vertex PointVaryings VertexPointVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MainDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(vertex_id, vertex_count - 1u);
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + idx]) :
        0u;

    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    const float3 local_pos = scene.GetLocalPosition(draw, idx);
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, local_pos));

    PointVaryings out;
    out.Color = point_color(scene, draw, element_state);
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    out.Position.z -= NdcOffsetFactor(scene) * 1.5f; // Push points in front of lines and faces.
    // Excite mode shows selected and active vertices only.
    out.PointSize = view.InteractionMode == InteractionMode_Excite && !is_selected && !is_active ? 0.0f : PointSize;
    return out;
}

fragment float4 VertexPointFragment(PointVaryings in [[stage_in]], float2 point_coord [[point_coord]]) {
    if (length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    return in.Color;
}

#endif
