#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

// Lay down the transmission prepass as the scene's plain-opaque and background pixels.
// The prepass holds unexposed radiance premultiplied by its coverage alpha, so exposing is a
// straight multiply and the pipeline blends premultiplied.
void main() {
    const vec4 prepass = textureLod(Samplers[nonuniformEXT(SceneViewUBO.TransmissionFramebufferSamplerSlot)], TexCoord, 0.0);
    OutColor = vec4(prepass.rgb * SceneViewUBO.Exposure, prepass.a);
}
