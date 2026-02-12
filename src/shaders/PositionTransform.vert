#version 450

#include "Bindless.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

layout(location = 1) flat out uint ObjectId;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.Index.Slot].Indices[draw.Index.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];
    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;

    vec3 normal = vert.Normal;
    const vec3 local_pos = ApplyArmatureDeform(draw, vert.Position, idx, normal);
    vec3 world_pos = vec3(world.M * vec4(local_pos, 1.0));
    if (should_apply_pending_transform(draw, idx)) {
        world_pos = apply_pending_transform(local_pos, world_pos, world, draw);
    }

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
}
