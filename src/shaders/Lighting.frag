#version 450

#include "SceneUBO.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) flat in uint FaceOverlayFlags;

layout(location = 0) out vec4 OutColor;

// Assumes `direction` and `normal` are normalized.
vec3 DirectionalLighting(vec3 direction, vec3 normal, vec3 color, float ambient, float intensity) {
    const vec3 diffuse_lighting = max(dot(normal, -direction), 0.0) * color * intensity;
    const vec3 ambient_lighting = color * ambient;
    return diffuse_lighting + ambient_lighting;
}

void main() {
    const vec3 normal = faceforward(WorldNormal, WorldPosition - SceneViewUBO.CameraPosition, WorldNormal);
    const vec3 view_lighting = DirectionalLighting(normalize(WorldPosition - SceneViewUBO.CameraPosition), normal, SceneViewUBO.ViewColor, SceneViewUBO.AmbientIntensity, 1);
    const vec3 directional_lighting = DirectionalLighting(SceneViewUBO.LightDirection, normal, SceneViewUBO.DirectionalColor, 0, SceneViewUBO.DirectionalIntensity);
    const vec3 lighting = view_lighting + directional_lighting;
    vec3 color = InColor.rgb * lighting;
    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionModeEdit &&
            SceneViewUBO.EditElement == EditElementFace;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ?
            mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) :
            selected.rgb;
        color = mix(color, overlay, selected.a);
    }
    OutColor = vec4(color, InColor.a);
}
