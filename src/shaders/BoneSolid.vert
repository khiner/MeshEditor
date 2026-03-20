#version 450

#include "Bindless.glsl"
#include "BoneUtils.glsl"

layout(location = 0) out vec4 Color;
layout(location = 1) flat out int Inverted;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    const mat4 M = trs_to_mat4(world);
    const vec3 world_pos = (M * vec4(vert.Position, 1.0)).xyz;
    const vec3 view_normal = SceneViewUBO.ViewRotation * normalize(mat3(M) * vert.Normal);
    const vec3 wire_color = bone_wire_color(load_bone_instance_state(draw));

    // Bone colors: solid base + state hint
    const vec3 bone_color = ViewportTheme.Colors.BoneSolid;
    const vec3 state_color = wire_color * wire_color * 0.1; // Blender's bone_hint_color_shade

    // Blender-style angled lighting
    const vec3 light = normalize(vec3(0.1, 0.1, 0.8));
    const float fac = clamp(dot(view_normal, light) * 0.8 + 0.2, 0.0, 1.0);
    const float alpha = (SceneViewUBO.BoneXRay != 0u) ? 0.6 : 1.0;
    Color = vec4(mix(state_color, bone_color, fac * fac), alpha);

    // Detect mirrored (negative-scale) instances for backface cull inversion
    Inverted = int(dot(cross(M[0].xyz, M[1].xyz), M[2].xyz) < 0.0);

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
}
