#version 450

layout(binding = 0) uniform sampler2D DepthSampler;
layout(location = 0) in vec2 TexCoord;

void main() {
    gl_FragDepth = texture(DepthSampler, TexCoord).r;
}
