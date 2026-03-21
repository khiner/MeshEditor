#version 450

// Billboard sphere vertex shader for bone joint rendering.
// Reads disc vertices from the vertex buffer, orients disc to face the camera,
// and passes sphere parameters for ray-tracing in the fragment shader.

#include "Bindless.glsl"
#include "BoneUtils.glsl"

layout(location = 0) flat out vec3 SphereCenter; // View-space sphere center
layout(location = 1) flat out uint ObjectId; // Read by SelectionFragment.frag
layout(location = 2) out vec3 ViewPos; // View-space position (for ray construction)
layout(location = 3) flat out vec4 BoneColor;
layout(location = 4) flat out vec4 StateColor;
layout(location = 5) flat out float SphereRadius; // View-space sphere radius

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;

    // Object mode: neutral shadow, no selection tint
    const bool is_object_mode = SceneViewUBO.InteractionMode == InteractionMode_Object;
    const vec3 bone_solid = ViewportTheme.Colors.BoneSolid;
    const vec3 hint_color = is_object_mode ? bone_solid : bone_joint_wire_color(load_bone_instance_state(draw));
    BoneColor = vec4(bone_solid, 1.0);
    StateColor = vec4(hint_color * hint_color * 0.1, 1.0);

    // Pass view-space data for fragment ray-tracing
    vec3 cam_pos = SceneViewUBO.CameraPosition;
    mat4 view = mat4(
        vec4(SceneViewUBO.ViewRotation[0], 0),
        vec4(SceneViewUBO.ViewRotation[1], 0),
        vec4(SceneViewUBO.ViewRotation[2], 0),
        vec4(-SceneViewUBO.ViewRotation * cam_pos, 1)
    );

    const BoneBillboard bb = bone_sphere_billboard(world, vert.Position);
    SphereCenter = (view * vec4(bb.center, 1.0)).xyz;
    SphereRadius = bb.radius;
    ViewPos = (view * vec4(bb.world_pos, 1.0)).xyz;

    gl_Position = SceneViewUBO.ViewProj * vec4(bb.world_pos, 1.0);
}
