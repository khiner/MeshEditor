#version 450

#include "SceneUBO.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;

layout(location = 0) out vec4 OutColor;

// Assumes `direction` and `normal` are normalized.
vec3 DirectionalLighting(vec3 direction, vec3 normal, vec3 color, float ambient, float intensity) {
    const vec3 diffuse_lighting = max(dot(normal, -direction), 0.0) * color * intensity;
    const vec3 ambient_lighting = color * ambient;
    return diffuse_lighting + ambient_lighting;
}

void main() {
    const vec3 view_lighting = DirectionalLighting(normalize(WorldPosition - SceneViewUBO.CameraPosition), WorldNormal, SceneViewUBO.ViewColor, SceneViewUBO.AmbientIntensity, 1);
    const vec3 directional_lighting = DirectionalLighting(SceneViewUBO.LightDirection, WorldNormal, SceneViewUBO.DirectionalColor, 0, SceneViewUBO.DirectionalIntensity);
    const vec3 lighting = view_lighting + directional_lighting;
    OutColor = vec4(InColor.rgb * lighting, InColor.a);
}
