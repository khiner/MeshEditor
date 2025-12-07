#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"

layout(set = 0, binding = 2) uniform sampler2D Samplers[];

layout(push_constant) uniform PC {
    int Manipulating;
    uint ObjectSamplerIndex;
} pc;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 EdgeColor;

const vec4 ManipulatingColor = vec4(1, 1, 1, 1);

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(Samplers[nonuniformEXT(pc.ObjectSamplerIndex)], 0));
    const float object_id = texelFetch(Samplers[nonuniformEXT(pc.ObjectSamplerIndex)], texel, 0).r;
    if (object_id < 1) {
        discard;
    } else {
        EdgeColor = bool(pc.Manipulating) ? ManipulatingColor : object_id == 1 ? Scene.SilhouetteActive : Scene.SilhouetteSelected;
    }
}
