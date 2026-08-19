#ifndef OBJECTEXTRAS_MSL
#define OBJECTEXTRAS_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "ObjectExtrasTransform.metal"
#include "Varyings.metal"
#include "MainDrawPushConstants.metal"

vertex LineVaryings ObjectExtrasVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MainDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    const float3 world_pos = ObjectExtrasWorldPos(scene, draw, vert, world, idx);

    const uint instance_state = draw.InstanceStateSlot != INVALID_SLOT ?
        uint(scene.InstanceStates(draw.InstanceStateSlot)[draw.FirstInstance]) :
        0u;

    const bool is_edit_mode = view.InteractionMode == InteractionMode_Edit;
    const float4 wire_color = is_edit_mode ? float4(float3(colors.WireEdit), 1.0f) : float4(float3(colors.Wire), 1.0f);
    const bool is_selected = (instance_state & STATE_SELECTED) != 0u;
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;

    float4 final_color = wire_color;
    if (is_selected) final_color = float4(float3(colors.ObjectSelected), 1.0f);
    if (is_selected && is_active) final_color = float4(float3(colors.ObjectActive), 1.0f);

    // Ground line and diamond: a fixed theme color, unaffected by selection state.
    if (GetVertexClass(scene, draw, idx) == VCLASS_GROUNDPOINT) final_color = float4(colors.Light);

    LineVaryings out;
    out.Color = final_color;
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    const float2 screen_pos = clip_to_frag_co(out.Position, float2(view.ViewportSize));
    out.EdgeStart = screen_pos;
    out.EdgePos = screen_pos;
    return out;
}

vertex ObjectIdVaryings ObjectExtrasSelectionVertex(
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
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    ObjectIdVaryings out;
    out.ObjectId = draw.ObjectIdSlot != INVALID_SLOT ? scene.ObjectIds(draw.ObjectIdSlot)[draw.FirstInstance] : 0u;
    out.Position = scene.ViewProj() * float4(ObjectExtrasWorldPos(scene, draw, vert, world, idx), 1.0f);
    out.PointSize = PointSize;
    return out;
}

#endif
