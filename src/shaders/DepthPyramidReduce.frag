#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Writes the farthest source depth covering the destination texel: one mip of the occlusion
// pyramid. An extra row or column covers odd source dimensions.

#include "DepthPyramidReducePushConstants.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) out float MaxDepth;

void main() {
    const int lod = int(pc.SrcLod);
    const ivec2 src_size = textureSize(Samplers[pc.SrcSamplerSlot], lod);
    const ivec2 base = ivec2(gl_FragCoord.xy) * 2;
    const ivec2 count = ivec2(2) + ivec2(src_size.x & 1, src_size.y & 1);
    float max_depth = 0.0;
    for (int y = 0; y < count.y; ++y) {
        for (int x = 0; x < count.x; ++x) {
            const ivec2 p = min(base + ivec2(x, y), src_size - 1);
            max_depth = max(max_depth, texelFetch(Samplers[pc.SrcSamplerSlot], p, lod).r);
        }
    }
    MaxDepth = max_depth;
}
