#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "BindlessBindings.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];
layout(push_constant) uniform PC {
    uint HdrSamplerSlot;
    uint ApplyTonemap;
};
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec3 hdr = texture(Samplers[HdrSamplerSlot], TexCoord).rgb;
    // In PBR modes, apply tone mapping + sRGB gamma. In Solid/Wireframe, pass through unchanged
    // (Lighting.frag outputs display-calibrated values, as in Blender's workbench engine).
    OutColor = ApplyTonemap != 0u
        ? vec4(linearTosRGB(toneMapPBRNeutral(hdr)), 1.0)
        : vec4(hdr, 1.0);
}
