#ifndef POSITIONTRANSFORM_MSL
#define POSITIONTRANSFORM_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "MainDrawPushConstants.metal"

vertex ObjectIdVaryings PositionTransformVertex(
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
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    ObjectIdVaryings out;
    out.ObjectId = draw.ObjectIdSlot != INVALID_SLOT ? scene.ObjectIds(draw.ObjectIdSlot)[draw.FirstInstance] : 0u;
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx)));
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    out.PointSize = PointSize;
    return out;
}

#endif
