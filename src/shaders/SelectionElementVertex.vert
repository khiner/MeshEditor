#version 450

#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.Index.Slot].Indices[draw.Index.Offset + uint(gl_VertexIndex)];

    // If ElementState.Slot is set, filter to only selected vertices.
    if (draw.ElementState.Slot != INVALID_SLOT) {
        const uint state = uint(ElementStateBuffers[draw.ElementState.Slot].States[draw.ElementState.Offset + idx]);
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

    gl_Position = SceneViewUBO.ViewProj * (world.M * vec4(vert.Position, 1.0));
    gl_PointSize = 6.0;
}
