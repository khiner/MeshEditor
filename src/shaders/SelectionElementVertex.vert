#version 450

#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;

void main() {
    const uint idx = IndexBuffers[pc.IndexSlot].Indices[pc.IndexOffset + uint(gl_VertexIndex)];

    // If ElementStateSlot is set, filter to only selected vertices.
    if (pc.ElementStateSlot != INVALID_SLOT) {
        const uint state = ElementStateBuffers[pc.ElementStateSlot].States[idx];
        if ((state & STATE_SELECTED) == 0u) {
            // Clip non-selected vertices by placing them outside the frustum.
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            gl_PointSize = 0.0;
            ElementId = 0u;
            return;
        }
    }

    const Vertex vert = VertexBuffers[pc.VertexSlot].Vertices[idx + pc.VertexOffset];
    const WorldMatrix world = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex];

    ElementId = pc.ElementIdOffset + idx + 1;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
