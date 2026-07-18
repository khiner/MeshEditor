#version 450

// Draws one bounding-box wireframe per instance, reading the instance-arena slot list, local
// bounds, and transform directly. 24 vertices per instance, two per box edge.
#define BINDLESS_CUSTOM_DRAW_PASS_PC 1
#define BINDLESS_NO_DRAW_LOOKUP 1
#include "BoundsBoxPushConstants.glsl"
#include "Bindless.glsl"
#include "AABB.glsl"

layout(location = 2) out vec4 Color;
layout(location = 12) flat out vec2 EdgeStart;
layout(location = 13) out vec2 EdgePos;

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer BoundsBuffer {
    AABB Bounds[];
} BoundsBuffers[];
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer BoxSlotBuffer {
    uint Slots[];
} BoxSlotBuffers[];

// Corner indices for the 12 box edges. Corner bits select Min/Max per axis: x=1, y=2, z=4.
const uint EdgeCorners[24] = uint[](
    0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u, // bottom ring
    4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u, // top ring
    0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u // verticals
);

void main() {
    const uint slot = BoxSlotBuffers[pc.SlotsSlot].Slots[gl_InstanceIndex];
    const AABB aabb = BoundsBuffers[pc.BoundsSlot].Bounds[slot];
    Color = vec4(0);
    EdgeStart = vec2(0);
    EdgePos = vec2(0);
    if (any(greaterThan(aabb.Min, aabb.Max))) {
        gl_Position = vec4(2, 2, 2, 1); // Empty bounds: place the vertex outside the clip volume.
        return;
    }
    const uint corner = EdgeCorners[gl_VertexIndex];
    const vec3 local = mix(aabb.Min, aabb.Max, vec3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u)));
    const Transform world = ModelBuffers[pc.ModelSlot].Models[slot];
    const vec3 world_pos = trs_transform_point(world, local);

    // Boxes draw only for selected instances.
    const uint instance_state = uint(InstanceStateBuffers[pc.StateSlot].States[slot]);
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
    Color = vec4(is_active ? ViewportTheme.Colors.ObjectActive : ViewportTheme.Colors.ObjectSelected, 1.0);

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    const vec2 screen_pos = (gl_Position.xy / gl_Position.w * 0.5 + 0.5) * SceneViewUBO.ViewportSize;
    EdgeStart = screen_pos;
    EdgePos = screen_pos;
}
