#pragma once

#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>

#include <optional>
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

// Levels are consecutive powers of two from Off->1 through X16->16.
constexpr float ToMaxAnisotropy(AnisotropicFilterLevel level) { return float(1u << unsigned(level)); }

// Motion blur, applied in Material Preview and Rendered while playing or scrubbing.
// Each step renders the scene once and blurs it along its own screen motion, so a single step covers the whole shutter.
// More steps subdivide the shutter and average the results.
// Shutter is the time in frames between shutter open and close, centered on the frame.
struct MotionBlur {
    float Shutter{0.5f};
    uint8_t Steps{1};
    float BleedingBias{100.f};
};

// Changes require command-buffer recording.
struct ViewportDisplay {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid};
    vec4 ClearColor{0.25f, 0.25f, 0.25f, 1.f};
    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowTetWireframe{false};
    bool ShowExtras{true}, ShowBones{true}, ShowOrigins{true}, ShowOutlineSelected{true};
    bool ShowOverlays{true};
    uint8_t NormalOverlays{0};
    // Screen-space error budget for the cluster LOD cut, in pixels. Zero renders original geometry alone.
    float LodErrorPixels{1.f};
    DebugChannel DebugChannel{DebugChannel::None};
    AnisotropicFilterLevel AnisotropicFilter{AnisotropicFilterLevel::X16};
    std::optional<MotionBlur> MotionBlur;
};

constexpr MotionBlur EffectiveMotionBlur(const ViewportDisplay &d) { return d.MotionBlur.value_or(MotionBlur{}); }
constexpr uint32_t MotionBlurSteps(const ViewportDisplay &d) { return std::max(1u, uint32_t(EffectiveMotionBlur(d).Steps)); }

struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Sample a mipmapped scene framebuffer at the refracted exit point for transmission.
    bool RealTransmission{true};
    // Exposure in EV stops. Scales linear color by 2^EV before tone mapping.
    float ExposureEV{0.f};
};

struct MaterialPreviewLighting : PBRViewportLighting {};
struct RenderedLighting : PBRViewportLighting {};

// The active studio HDRI environment, by source name so it stays stable across runs (unlike the directory-scan index).
struct StudioEnvironment {
    std::string Name;
};

const PBRViewportLighting &GetActivePbrLighting(const entt::registry &, entt::entity viewport, ViewportShadingMode);

struct ViewportExtent {
    uvec2 Value{};
};
