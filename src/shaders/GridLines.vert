#version 450

#include "SceneUBO.glsl"

layout(location = 0) out vec4 PlanePos;

// The grid plane y = 0: three triangles fanned around the origin with outer vertices at infinity
// (w = 0), so rasterization supplies the plane's depth and the early depth test skips hidden fragments.
const float R3 = 0.86602540378; // sin(120 degrees)
// Outer directions 120 degrees apart: each triangle covers one third of the plane.
const vec4 Verts[9] = vec4[](
    vec4(0, 0, 0, 1), vec4(1, 0, 0, 0), vec4(-0.5, 0, R3, 0),
    vec4(0, 0, 0, 1), vec4(-0.5, 0, R3, 0), vec4(-0.5, 0, -R3, 0),
    vec4(0, 0, 0, 1), vec4(-0.5, 0, -R3, 0), vec4(1, 0, 0, 0)
);

void main() {
    PlanePos = Verts[gl_VertexIndex];
    gl_Position = SceneViewUBO.ViewProj * PlanePos;
}
