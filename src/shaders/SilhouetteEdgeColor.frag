#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "BindlessBindings.glsl"
#include "SceneUBO.glsl"
#include "SilhouetteEdgeColorPushConstants.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(push_constant) uniform PC {
    SilhouetteEdgeColorPushConstants Data;
} pc;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 EdgeColor;

const vec4 ManipulatingColor = vec4(1, 1, 1, 1);

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(Samplers[pc.Data.ObjectSamplerIndex], 0));
    const uint object_id = uint(texelFetch(Samplers[pc.Data.ObjectSamplerIndex], texel, 0).r);
    if (object_id == 0) {
        discard;
    } else {
        const bool is_active = pc.Data.ActiveObjectId != 0u && object_id == pc.Data.ActiveObjectId;
        EdgeColor = bool(pc.Data.Manipulating) ? ManipulatingColor : is_active ? ViewportTheme.Colors.ObjectActive : ViewportTheme.Colors.ObjectSelected;
    }
}
