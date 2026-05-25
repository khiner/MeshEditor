#pragma once

#include "gpu/DebugChannel.h"

#include <entt/entity/registry.hpp>

#include <cstdint>
#include <vulkan/vulkan.hpp>

enum class ViewportShadingMode : uint8_t {
    Wireframe,
    Solid,
    MaterialPreview,
    Rendered,
};

// Component on the viewport singleton entity. Changes require command buffer re-recording.
struct ViewportDisplay {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid}; // last non-wireframe mode (for Shift+Z toggle)
    vk::ClearColorValue ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowTetWireframe{false};
    bool ShowExtras{true}, ShowBones{true}, ShowOrigins{true}, ShowOutlineSelected{true};
    bool ShowOverlays{true}; // Master toggle for all overlays
    uint8_t NormalOverlays{0}; // Bitmask of Element
    DebugChannel DebugChannel{DebugChannel::None};
};

// Scene lights/world toggles + studio env controls
struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Render the scene into a transmission framebuffer (with mips) and sample it at the
    // refracted ray exit point, instead of approximating refraction by sampling the IBL.
    bool RealTransmission{true};
};

// Two distinct ECS component types sharing the same layout, with different defaults
struct MaterialPreviewLighting : PBRViewportLighting {}; // defaults: both OFF (studio HDRI)
struct RenderedLighting : PBRViewportLighting {}; // defaults: both ON (scene world/lights)

inline const PBRViewportLighting &GetActivePbrLighting(const entt::registry &r, entt::entity viewport, ViewportShadingMode mode) {
    return mode == ViewportShadingMode::Rendered ? static_cast<const PBRViewportLighting &>(r.get<const RenderedLighting>(viewport)) : static_cast<const PBRViewportLighting &>(r.get<const MaterialPreviewLighting>(viewport));
}

struct ViewportExtent {
    vk::Extent2D Value{};
};
