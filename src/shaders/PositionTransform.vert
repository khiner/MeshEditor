#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

layout(location = 1) flat out uint ObjectId;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];
    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;

    vec3 normal = vert.Normal;
    vec3 morphed_pos = vert.Position;
    ApplyMorphDeform(draw, morphed_pos, idx, normal);
    const vec3 local_pos = ApplyArmatureDeform(draw, morphed_pos, idx, normal);
    vec3 world_pos = apply_pending_transform(draw, world, local_pos, idx);

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    gl_PointSize = 6.0;
}
