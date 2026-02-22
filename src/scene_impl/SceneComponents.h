#pragma once

#include "gpu/SceneViewUBO.h"
#include "gpu/ViewportTheme.h"

namespace {
// Component on the scene singleton entity. Sent directly as UBO to shader.

// Component on the scene singleton entity. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid}; // last non-wireframe mode (for Shift+Z toggle)
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    FaceColorMode FaceColorMode{FaceColorMode::Mesh};
    bool SmoothShading{false}, ShowGrid{true}, ShowBoundingBoxes{false};
    uint8_t NormalOverlays{0}; // Bitmask of he::Element
};

// Shared PBR viewport lighting controls (Blender-style scene lights/world toggles + studio env controls).
struct PBRViewportLighting {
    bool UseSceneLights;
    bool UseSceneWorld;
    float EnvIntensity;
    float EnvRotationDegrees;
};

// Two distinct ECS component types sharing the same layout, with different defaults.
struct MaterialPreviewLighting : PBRViewportLighting {}; // defaults: both OFF (studio HDRI)
struct RenderedLighting : PBRViewportLighting {}; // defaults: both ON (scene world/lights)

struct SceneInteraction {
    InteractionMode Mode{InteractionMode::Object};
};

struct SceneEditMode {
    he::Element Value{he::Element::Vertex};
};

struct ViewportExtent {
    vk::Extent2D Value{};
};
} // namespace
