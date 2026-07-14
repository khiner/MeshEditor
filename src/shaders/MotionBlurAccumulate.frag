#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint ColorSamplerSlot;
} pc;

// Decode the display-referred sub-frame back to scene-linear so accumulation (additive blend) averages in linear.
void main() {
    const vec3 color = texture(Samplers[pc.ColorSamplerSlot], TexCoord).rgb;
    OutColor = vec4(displayToLinear(color), 1.0);
}
