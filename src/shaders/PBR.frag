#version 450

// Adapted from Khronos glTF-Sample-Renderer shader logic, pulled 2026-02-16.
//  - hardcoded material defaults instead of full material graph/textures
//  - loop over bindless LightBuffer (LightCount/LightSlot)
//  - MeshEditor face-overlay behavior
//  - no IBL/material extensions yet (lighting-first validation)

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#include "SceneUBO.glsl"
#include "BindlessBindings.glsl"
#include "PunctualLight.glsl"
#include "WorldTransform.glsl"
#include "TRSUtils.glsl"
#include "brdf.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_LightBuffer, scalar) readonly buffer LightBufferBlock {
    PunctualLight Lights[];
} LightBuffers[];

layout(set = 0, binding = BINDING_ModelBuffer, scalar) readonly buffer ModelBuffer {
    WorldTransform Models[];
} ModelBuffers[];

#include "punctual.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) flat in uint FaceOverlayFlags;

layout(location = 0) out vec4 OutColor;

void main() {
    const vec3 N = normalize(WorldNormal);
    const vec3 V = normalize(SceneViewUBO.CameraPosition - WorldPosition);
    const float NdotV = max(dot(N, V), 1e-4);

    const vec3 baseColor = InColor.rgb * 0.7;
    const float metallic = 0.0;
    const float perceptualRoughness = 0.5;
    const float alphaRoughness = perceptualRoughness * perceptualRoughness;

    const vec3 f0 = mix(vec3(0.04), baseColor, metallic);
    const vec3 f90 = vec3(1.0);
    const vec3 diffuseColor = baseColor * (1.0 - metallic);

    vec3 color = baseColor * 0.03; // Small ambient floor for unlit areas.
    for (uint i = 0u; i < SceneViewUBO.LightCount; ++i) {
        const PunctualLight light = LightBuffers[nonuniformEXT(SceneViewUBO.LightSlot)].Lights[i];

        vec3 L;
        const vec3 lightIntensity = getLightIntensity(light, WorldPosition, L);
        const float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        const vec3 H = normalize(V + L);
        const float VdotH = max(dot(V, H), 0.0);
        const float NdotH = max(dot(N, H), 0.0);

        const vec3 diffuse = BRDF_lambertian(diffuseColor);
        const vec3 specular = BRDF_specularGGX(f0, f90, alphaRoughness, VdotH, NdotL, NdotV, NdotH);
        color += (diffuse + specular) * lightIntensity * NdotL;
    }

    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionModeEdit &&
            SceneViewUBO.EditElement == EditElementFace;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ?
            mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) :
            selected.rgb;
        color = mix(color, overlay, selected.a);
    }

    color = toneMapPBRNeutral(max(color, vec3(0.0)));
    color = linearTosRGB(color);
    OutColor = vec4(color, InColor.a);
}
