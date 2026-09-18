#pragma once

#include "numeric/uvec2.h"

#include "gpu/DebugChannel.h"
#include "gpu/InteractionMode.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include "state/Entity.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

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

enum class MotionBlurMethod : uint8_t {
    Fast,
    FullSampling,
};

struct MotionBlur {
    float Shutter{0.5f};
    uint8_t Steps{16};
    MotionBlurMethod Method{MotionBlurMethod::Fast};
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
    bool XRaySolid{false}, XRayWireframe{true};
    float XRayAlpha{0.5f}, XRayAlphaWireframe{0.f};
};

// Wireframe and Solid shade with the workspace lights and support X-ray display.
constexpr bool WorkbenchShading(ViewportShadingMode mode) { return mode == ViewportShadingMode::Wireframe || mode == ViewportShadingMode::Solid; }
// The X-ray flag of the current shading family. Edit-mode element selection follows it directly.
constexpr bool XRayFlag(const ViewportDisplay &d) { return d.ViewportShading == ViewportShadingMode::Wireframe ? d.XRayWireframe : d.XRaySolid; }
constexpr float XRayOpacity(const ViewportDisplay &d) { return d.ViewportShading == ViewportShadingMode::Wireframe ? d.XRayAlphaWireframe : d.XRayAlpha; }
// X-ray display: the flag is set, the opacity is below one, and the shading is workbench.
constexpr bool XRayActive(const ViewportDisplay &d) { return WorkbenchShading(d.ViewportShading) && XRayFlag(d) && XRayOpacity(d) < 1.f; }
// Strength of overlays behind the scene surface. Zero leaves them to the depth test.
// X-ray display fades them by its opacity, and Edit mode with the flag draws back wires at half strength, as in Blender.
constexpr float OverlayBehindOpacity(const ViewportDisplay &d, InteractionMode mode) {
    if (XRayActive(d)) return 1.f - XRayOpacity(d);
    return mode == InteractionMode::Edit && XRayFlag(d) ? 0.5f : 0.f;
}

constexpr MotionBlur EffectiveMotionBlur(const ViewportDisplay &d) { return d.MotionBlur.value_or(MotionBlur{}); }
constexpr uint32_t MotionBlurSteps(const ViewportDisplay &d) { return std::clamp(uint32_t(EffectiveMotionBlur(d).Steps), 1u, 64u); }

struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Sample a mipmapped scene framebuffer at the refracted exit point for transmission.
    bool RealTransmission{true};
    // Exposure in EV stops. Scales linear color by 2^EV before tone mapping.
    float ExposureEV{0.f};
};

struct MaterialPreviewLighting {
    PBRViewportLighting Value;
};
struct RenderedLighting {
    PBRViewportLighting Value;
};

// The active studio HDRI environment, by source name so it stays stable across runs (unlike the directory-scan index).
struct StudioEnvironment {
    std::string Name;
};

const PBRViewportLighting &GetActivePbrLighting(const state::Scene &, state::Entity viewport, ViewportShadingMode);

struct ViewportExtent {
    uvec2 Value{};
};
