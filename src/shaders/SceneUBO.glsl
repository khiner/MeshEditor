#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"

layout(set = 0, binding = BINDING_SceneViewUBO, scalar) uniform SceneViewUBO {
    mat4 View;
    mat4 Proj;
    vec3 CameraPosition;
    float CameraNear;
    float CameraFar;
    vec3 Pad0;
    vec3 ViewColor;
    float AmbientIntensity;
    vec3 DirectionalColor;
    float DirectionalIntensity;
    vec3 LightDirection;
    float Pad1;
} SceneView;

layout(set = 0, binding = BINDING_ViewportThemeUBO, scalar) uniform ViewportThemeUBO {
    vec4 SilhouetteActive;
    vec4 SilhouetteSelected;
    vec4 EdgeColor;
    vec4 FaceNormalColor;
    vec4 VertexNormalColor;
    vec4 VertexUnselectedColor;
    vec4 SelectedColor;
    vec4 ActiveColor;
} ViewportTheme;
