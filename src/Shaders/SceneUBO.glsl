// Shared SceneUBO uniform block definition
// Include this after declaring #version and #extension GL_EXT_nonuniform_qualifier

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 View;
    mat4 Proj;
    vec4 CameraPositionNear; // xyz: camera, w: near
    vec4 ViewColorAndAmbient;
    vec4 DirectionalColorAndIntensity;
    vec4 LightDirectionFar; // xyz: dir, w: far
    vec4 SilhouetteActive;
    vec4 SilhouetteSelected;
} Scene;
