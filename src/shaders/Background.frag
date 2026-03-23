#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"
#include "BindlessBindings.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_CubeSampler) uniform samplerCube CubeSamplers[];

layout(location = 0) in vec2 InNDC;
layout(location = 0) out vec4 OutColor;

mat3 EnvRotation() {
    const float s = sin(SceneViewUBO.EnvRotationRadians);
    const float c = cos(SceneViewUBO.EnvRotationRadians);
    return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

void main() {
    if (SceneViewUBO.WorldOpacity <= 0.0 || SceneViewUBO.Ibl.SpecularEnvSamplerSlot == 0xFFFFFFFFu) discard;

    const mat3 proj3 = mat3(SceneViewUBO.ViewProj) * transpose(SceneViewUBO.ViewRotation);
    const mat3 inv_rot = transpose(SceneViewUBO.ViewRotation);
    const vec3 world_dir = normalize(inv_rot * vec3(InNDC.x / proj3[0][0], InNDC.y / proj3[1][1], -1.0));
    const vec3 env_dir = EnvRotation() * world_dir;
    const uint mip_count = max(SceneViewUBO.Ibl.SpecularEnvMipCount, 1u);
    const float lod = clamp(SceneViewUBO.BackgroundBlur, 0.0, 1.0) * float(mip_count - 1u);
    const vec3 linear = textureLod(
        CubeSamplers[nonuniformEXT(SceneViewUBO.Ibl.SpecularEnvSamplerSlot)], env_dir, lod
    ).rgb * SceneViewUBO.EnvIntensity;
    OutColor = vec4(linearTosRGB(toneMapPBRNeutral(linear)), SceneViewUBO.WorldOpacity);
}
