#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"
#include "SilhouetteEdgeColorPushConstants.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 EdgeColor;

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(Samplers[pc.ObjectSamplerIndex], 0));
    const uint object_id = uint(texelFetch(Samplers[pc.ObjectSamplerIndex], texel, 0).r);
    if (object_id == 0) discard;

    // ActiveObjectId == UINT32_MAX is a sentinel meaning "all objects are active" (used for armatures,
    // which have no single ObjectId of their own — all bone instance pixels should show as active).
    const bool is_active = pc.ActiveObjectId == 0xFFFFFFFFu || (pc.ActiveObjectId != 0u && object_id == pc.ActiveObjectId);
    EdgeColor = bool(pc.Manipulating) ? vec4(ViewportTheme.Colors.Transform, 1.0) :
        is_active ? vec4(ViewportTheme.Colors.ObjectActive, 1.0) :
                    vec4(ViewportTheme.Colors.ObjectSelected, 1.0);
}
