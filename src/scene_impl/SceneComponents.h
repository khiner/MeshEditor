#pragma once

#include "generated/SceneViewUBO.h"

namespace {

struct ViewportThemeColors {
    // These mirror Blender: Preferences->Themes->3D Viewport->...
    vec4 Wire{0, 0, 0, 1}; // Wire
    vec4 WireEdit{0, 0, 0, 1}; // Wire Edit
    vec4 ObjectActive{1, 0.627, 0.157, 1}; // Active Object
    vec4 ObjectSelected{0.929, 0.341, 0, 1}; // Object Selected
    vec4 Vertex{0, 0, 0, 1}; // Vertex
    vec4 ElementSelected{1, 0.478, 0, 1}; // Vertex Select
    vec4 ElementActive{1, 1, 1, 1}; // Active Vertex/Edge/Face
    vec4 FaceNormal{0.133, 0.867, 0.867, 1}; // Face Normal
    vec4 VertexNormal{0.137, 0.380, 0.867, 1}; // Vertex Normal
};

// Component on the scene singleton entity. Sent directly as UBO to shader.
struct ViewportTheme {
    ViewportThemeColors Colors{};
    uint32_t SilhouetteEdgeWidth{1};
};

// Component on the scene singleton entity. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    FaceColorMode FaceColorMode{FaceColorMode::Mesh};
    bool SmoothShading{false}, ShowGrid{true}, ShowBoundingBoxes{false};
    uint8_t NormalOverlays{0}; // Bitmask of he::Element
};

struct SceneInteraction {
    InteractionMode Mode{InteractionMode::Object};
};

struct SceneEditMode {
    he::Element Value{he::Element::Face};
};

struct ViewportExtent {
    vk::Extent2D Value{};
};
} // namespace
