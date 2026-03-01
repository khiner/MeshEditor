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

    const mat4 inv_vp = inverse(SceneViewUBO.ViewProj);
    const vec4 far_world = inv_vp * vec4(InNDC, 1.0, 1.0);
    const vec3 world_dir = normalize(far_world.xyz / far_world.w - SceneViewUBO.CameraPosition);
    const vec3 env_dir = EnvRotation() * world_dir;
    const uint mip_count = max(SceneViewUBO.Ibl.SpecularEnvMipCount, 1u);
    const float lod = clamp(SceneViewUBO.BackgroundBlur, 0.0, 1.0) * float(mip_count - 1u);
    const vec3 linear = textureLod(
        CubeSamplers[nonuniformEXT(SceneViewUBO.Ibl.SpecularEnvSamplerSlot)], env_dir, lod
    ).rgb * SceneViewUBO.EnvIntensity;
    OutColor = vec4(linearTosRGB(toneMapPBRNeutral(linear)), SceneViewUBO.WorldOpacity);
}
