#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "BindlessBindings.glsl"
#include "SceneUBO.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(push_constant) uniform PC {
    uint Manipulating;
    uint ObjectSamplerIndex;
    uint ActiveObjectId;
} pc;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 EdgeColor;

const vec4 ManipulatingColor = vec4(1, 1, 1, 1);

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(Samplers[pc.ObjectSamplerIndex], 0));
    const uint object_id = uint(texelFetch(Samplers[pc.ObjectSamplerIndex], texel, 0).r);
    if (object_id == 0) {
        discard;
    } else {
        const bool is_active = pc.ActiveObjectId != 0u && object_id == pc.ActiveObjectId;
        EdgeColor = bool(pc.Manipulating) ? ManipulatingColor : is_active ? Scene.SilhouetteActive : Scene.SilhouetteSelected;
    }
}
