#version 450

layout(binding = 0) uniform sampler2D DepthSampler;
layout(location = 0) in vec2 TexCoord;

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(DepthSampler, 0));
    gl_FragDepth = texelFetch(DepthSampler, texel, 0).r;
}
