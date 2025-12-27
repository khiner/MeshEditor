#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const uint idx = IndexBuffers[nonuniformEXT(pc.IndexSlot)].Indices[gl_VertexIndex];
    const Vertex vert = VertexBuffers[nonuniformEXT(pc.VertexSlot)].Vertices[idx];
    const WorldMatrix world = ModelBuffers[nonuniformEXT(pc.ModelSlot)].Models[pc.FirstInstance + gl_InstanceIndex];

    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    uint state = 0u;
    vec4 base_color = pc.LineColor;
    if (pc.ObjectIdSlot != INVALID_SLOT) {
        const uint element_id = ObjectIdBuffers[nonuniformEXT(pc.ObjectIdSlot)].Ids[idx];
        if (pc.ElementStateSlot != INVALID_SLOT && element_id != 0u) {
            state = ElementStateBuffers[nonuniformEXT(pc.ElementStateSlot)].States[element_id - 1u];
        }
        base_color = Scene.BaseColor;
    } else if (pc.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[nonuniformEXT(pc.ElementStateSlot)].States[idx];
        base_color = Scene.EdgeColor;
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    Color = is_active ? Scene.ActiveColor : is_selected ? Scene.SelectedColor : base_color;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
}
