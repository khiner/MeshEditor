#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"
#include "SceneViewUBO.glsl"

const uint InteractionModeObject = 0u, InteractionModeEdit = 1u, InteractionModeExcite = 2u;

layout(set = 0, binding = BINDING_SceneViewUBO, scalar) uniform SceneViewUBOBlock {
    SceneViewUBO SceneView;
};

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
