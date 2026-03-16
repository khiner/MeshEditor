#version 450

// Billboard sphere outline for bone joints.
// Renders the disc ring as a line list, oriented to face the camera.

#include "Bindless.glsl"
#include "BoneUtils.glsl"

layout(location = 2) out vec4 Color;
layout(location = 12) flat out vec2 EdgeStart;
layout(location = 13) out vec2 EdgePos;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    Color = vec4(bone_joint_wire_color(load_bone_instance_state(draw)), 1.0);

    const BoneBillboard bb = bone_sphere_billboard(world, vert.Position);
    vec4 clip_pos = SceneViewUBO.ViewProj * vec4(bb.world_pos, 1.0);

    // Blender's approach: push outline 1px outward in screen space (away from sphere center)
    // so it never overlaps with the sphere fill at all — avoids all depth fighting.
    vec4 center_clip = SceneViewUBO.ViewProj * vec4(bb.center, 1.0);
    vec2 ofs_dir = normalize(clip_pos.xy / clip_pos.w - center_clip.xy / center_clip.w);
    clip_pos.xy += ofs_dir * (2.0 / SceneViewUBO.ViewportSize) * clip_pos.w;

    gl_Position = clip_pos;
    vec2 screen_pos = (clip_pos.xy / clip_pos.w * 0.5 + 0.5) * SceneViewUBO.ViewportSize;
    EdgeStart = screen_pos;
    EdgePos = screen_pos;
}
