#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"

const uint InteractionModeObject = 0u, InteractionModeEdit = 1u, InteractionModeExcite = 2u;

layout(set = 0, binding = BINDING_SceneViewUBO, scalar) uniform SceneViewUBO {
    mat4 View;
    mat4 Proj;
    vec3 CameraPosition;
    float CameraNear;
    float CameraFar;
    vec3 ViewColor;
    float AmbientIntensity;
    vec3 DirectionalColor;
    float DirectionalIntensity;
    vec3 LightDirection;
    uint InteractionMode;
} SceneView;

struct ViewportThemeColors {
    vec4 Wire;
    vec4 WireEdit;
    vec4 ObjectActive;
    vec4 ObjectSelected;
    vec4 Vertex;
    vec4 ElementSelected;
    vec4 ElementActive;
    vec4 FaceNormal;
    vec4 VertexNormal;
};

layout(set = 0, binding = BINDING_ViewportThemeUBO, scalar) uniform ViewportThemeUBO {
    ViewportThemeColors Colors;
    uint SilhouetteEdgeWidth;
} ViewportTheme;
