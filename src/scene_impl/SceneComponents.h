#pragma once

#include "gpu/SceneViewUBO.h"
#include "gpu/ViewportTheme.h"

namespace {
// Component on the scene singleton entity. Sent directly as UBO to shader.

// Component on the scene singleton entity. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    FaceColorMode FaceColorMode{FaceColorMode::Mesh};
    bool SmoothShading{false}, ShowGrid{true}, ShowBoundingBoxes{false};
    uint8_t NormalOverlays{0}; // Bitmask of he::Element
};

// Rendered viewport lighting controls (Blender-style scene lights/world toggles + studio env controls).
struct RenderedLighting {
    bool UseSceneLightsRender{true};
    bool UseSceneWorldRender{true};
    float EnvIntensity{1.f};
    float EnvRotationDegrees{0.f};
};

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
