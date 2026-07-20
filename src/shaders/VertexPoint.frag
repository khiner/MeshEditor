#version 450

layout(location = 0) in vec4 InColor;

layout(location = 0) out vec4 OutColor;

void main() {
    // Make points circular by discarding fragments outside a circle
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;

    OutColor = InColor;
}
