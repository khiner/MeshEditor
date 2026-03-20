#pragma once

// Component on the scene singleton entity. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid}; // last non-wireframe mode (for Shift+Z toggle)
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    FaceColorMode FaceColorMode{FaceColorMode::Mesh};
    bool SmoothShading{false}, ShowGrid{true}, ShowBoundingBoxes{false};
    uint8_t NormalOverlays{0}; // Bitmask of Element
};

// Shared PBR viewport lighting controls (Blender-style scene lights/world toggles + studio env controls).
struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
};

// Two distinct ECS component types sharing the same layout, with different defaults.
struct MaterialPreviewLighting : PBRViewportLighting {}; // defaults: both OFF (studio HDRI)
struct RenderedLighting : PBRViewportLighting {}; // defaults: both ON (scene world/lights)

struct SceneInteraction {
    InteractionMode Mode{InteractionMode::Object};
};

struct SceneEditMode {
    Element Value{Element::Vertex};
};

struct ViewportExtent {
    vk::Extent2D Value{};
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + Punctual bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};

// Snapshot of selection state at the start of a shift+box-drag.
// Presence on SceneEntity means an additive box-drag is active.
struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    std::vector<uint32_t> ElementBitset;
};
