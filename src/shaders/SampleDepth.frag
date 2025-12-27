#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 2) uniform sampler2D Samplers[];
layout(push_constant) uniform PC {
    uint DepthSamplerIndex;
} pc;
layout(location = 0) in vec2 TexCoord;

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(Samplers[pc.DepthSamplerIndex], 0));
    gl_FragDepth = texelFetch(Samplers[pc.DepthSamplerIndex], texel, 0).r;
}
