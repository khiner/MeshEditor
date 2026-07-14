#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint AccumSamplerSlot;
    float InvSamples; // 1 / sample count.
} pc;

// Average the accumulated scene-linear samples and re-apply the display transform.
void main() {
    const vec3 accum = texture(Samplers[pc.AccumSamplerSlot], TexCoord).rgb;
    OutColor = vec4(linearToDisplay(accum * pc.InvSamples), 1.0);
}
