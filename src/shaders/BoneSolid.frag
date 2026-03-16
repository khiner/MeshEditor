#version 450

layout(location = 0) in vec4 InColor;
layout(location = 1) flat in int Inverted;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData; // No line data for solid fill

void main() {
    // Manual backface cull: discard back-faces (accounting for mirrored instances)
    if ((Inverted == 1) == gl_FrontFacing) discard;

    OutColor = InColor;
    OutLineData = vec4(0); // Not a line
}
