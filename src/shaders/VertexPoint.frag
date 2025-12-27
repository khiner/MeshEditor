#version 450

layout(location = 0) in vec3 WorldNormal; // Unused
layout(location = 1) in vec3 WorldPosition; // Unused
layout(location = 2) in vec4 InColor;

layout(location = 0) out vec4 OutColor;

void main() {
    // Make points circular by discarding fragments outside a circle
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) {
        discard;
    }
    OutColor = InColor;
}
