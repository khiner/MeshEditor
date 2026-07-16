#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint AccumSamplerSlot;
    float InvSamples; // 1 / sample count.
} pc;

// Average the summed sub-frames. Color and coverage are both premultiplied, so both scale together.
void main() {
    OutColor = texture(Samplers[pc.AccumSamplerSlot], TexCoord) * pc.InvSamples;
}
