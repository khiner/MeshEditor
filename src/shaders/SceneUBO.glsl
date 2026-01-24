#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"
#include "SceneViewUBO.glsl"
#include "ViewportTheme.glsl"

const uint InteractionModeObject = 0u, InteractionModeEdit = 1u, InteractionModeExcite = 2u;

layout(set = 0, binding = BINDING_SceneViewUBO, scalar) uniform SceneViewUBOBlock {
    SceneViewUBO SceneView;
};

layout(set = 0, binding = BINDING_ViewportThemeUBO, scalar) uniform ViewportThemeUBOBlock {
    ViewportTheme Data;
} ViewportThemeUBO;

#define ViewportTheme ViewportThemeUBO.Data
