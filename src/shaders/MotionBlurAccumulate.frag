#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint GatherSamplerSlot;
} pc;

// Sum this step's blurred scene into the accumulation target (additive blend). The gather output is
// premultiplied, so alpha sums alongside color and a blurred edge keeps its partial coverage of the backdrop.
void main() {
    OutColor = texture(Samplers[pc.GatherSamplerSlot], TexCoord);
}
