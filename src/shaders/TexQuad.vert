#version 450

layout(location = 0) out vec2 TexCoord;

// Triangle strip.
const vec2 QuadPositions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

void main() {
    TexCoord = QuadPositions[gl_VertexIndex] * 0.5 + 0.5;
    gl_Position = vec4(QuadPositions[gl_VertexIndex], 0, 1);
}
