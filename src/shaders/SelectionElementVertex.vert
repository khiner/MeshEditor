#version 450

#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlot].Indices[draw.IndexOffset + uint(gl_VertexIndex)];

    // If ElementStateSlot is set, filter to only selected vertices.
    if (draw.ElementStateSlot != INVALID_SLOT) {
        const uint state = ElementStateBuffers[draw.ElementStateSlot].States[idx];
        if ((state & STATE_SELECTED) == 0u) {
            // Clip non-selected vertices by placing them outside the frustum.
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            gl_PointSize = 0.0;
            ElementId = 0u;
            return;
        }
    }

    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ElementId = draw.ElementIdOffset + idx + 1;
    gl_Position = SceneViewUBO.Proj * SceneViewUBO.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
