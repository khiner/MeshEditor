#version 450

#include "Bindless.glsl"
#include "TransformUtils.glsl"

layout(location = 1) flat out uint ObjectId;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];
    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;

    const vec3 world_pos = apply_object_pending_transform(draw, trs_transform_point(world, GetLocalPosition(draw, idx)));

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    gl_PointSize = 8.0;
}
