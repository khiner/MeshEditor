#version 450

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;

layout(location = 0) out vec4 OutColor;

layout(binding = 0) uniform CameraUBO {
    mat4 View;
    mat4 Proj;
    vec3 Position;
} Camera;

layout(binding = 1) uniform LightsUBO {
    vec4 ViewColorAndAmbient;
    vec4 DirectionalColorAndIntensity;
    vec3 Direction;
} Lights;

// Assumes `direction` and `normal` are normalized.
vec3 DirectionalLighting(vec3 direction, vec3 normal, vec3 color, float ambient, float intensity) {
    const vec3 diffuse_lighting = max(dot(normal, -direction), 0.0) * color * intensity;
    const vec3 ambient_lighting = color * ambient;
    return diffuse_lighting + ambient_lighting;
}

void main() {
    const vec3 view_lighting = DirectionalLighting(normalize(WorldPosition - Camera.Position), WorldNormal, Lights.ViewColorAndAmbient.rgb, Lights.ViewColorAndAmbient.a, 1);
    const vec3 directional_lighting = DirectionalLighting(Lights.Direction, WorldNormal, Lights.DirectionalColorAndIntensity.rgb, 0, Lights.DirectionalColorAndIntensity.a);
    const vec3 lighting = view_lighting + directional_lighting;
    OutColor = vec4(InColor.rgb * lighting, InColor.a);
}
