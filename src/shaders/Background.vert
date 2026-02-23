#version 450

layout(location = 0) out vec2 OutNDC;

// Triangle strip covering the full screen.
const vec2 Positions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

void main() {
    const vec2 p = Positions[gl_VertexIndex];
    OutNDC = p;
    gl_Position = vec4(p, 1.0, 1.0); // z=1 → far plane; geometry overdrawes via depth test
}
