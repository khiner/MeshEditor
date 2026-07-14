#pragma once

#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>

#include <string>

enum class ViewportShadingMode : uint8_t {
    Wireframe,
    Solid,
    MaterialPreview,
    Rendered,
};

enum class AnisotropicFilterLevel : uint8_t {
    Off,
    X2,
    X4,
    X8,
    X16
};

// Levels are consecutive powers of two: Off->1, X2->2, ... X16->16.
constexpr float ToMaxAnisotropy(AnisotropicFilterLevel level) { return float(1u << unsigned(level)); }

// Component on the viewport singleton entity. Changes require command buffer re-recording.
struct ViewportDisplay {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid}; // last non-wireframe mode (for Shift+Z toggle)
    vec4 ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowTetWireframe{false};
    bool ShowExtras{true}, ShowBones{true}, ShowOrigins{true}, ShowOutlineSelected{true};
    bool ShowOverlays{true}; // Master toggle for all overlays
    uint8_t NormalOverlays{0}; // Bitmask of Element
    DebugChannel DebugChannel{DebugChannel::None};
    AnisotropicFilterLevel AnisotropicFilter{AnisotropicFilterLevel::X16};
};

// Scene lights/world toggles + studio env controls
struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Render the scene into a transmission framebuffer (with mips) and sample it at the
    // refracted ray exit point, instead of approximating refraction by sampling the IBL.
    bool RealTransmission{true};
    // Exposure in EV stops. Scales linear color by 2^EV before tone mapping.
    float ExposureEV{0.f};
};

// Two distinct ECS component types sharing the same layout, with different defaults
struct MaterialPreviewLighting : PBRViewportLighting {}; // defaults: both OFF (studio HDRI)
struct RenderedLighting : PBRViewportLighting {}; // defaults: both ON (scene world/lights)

// The active studio HDRI environment, by source name so it stays stable across runs (unlike the directory-scan index).
struct StudioEnvironment {
    std::string Name;
};

// Editor display/lighting settings on the viewport entity.

const PBRViewportLighting &GetActivePbrLighting(const entt::registry &, entt::entity viewport, ViewportShadingMode);

// Logical (window) size of the viewport in pixels. Engine state, held as a ctx singleton.
struct ViewportExtent {
    uvec2 Value{};
};
