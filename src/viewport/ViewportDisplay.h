#pragma once

#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>

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

// Keep the nine-byte action layout: 0x80 marks fast blur, with its inactive sample count
// in the former bleeding-bias field. 0x81..0xc0 are full sample counts from older logs.
constexpr auto serialize(auto &archive, const MotionBlur &blur) {
    const uint8_t count = std::clamp<uint8_t>(blur.Steps, 1, 64);
    const uint8_t encoded = blur.Method == MotionBlurMethod::Fast ? 0x80u : 0x80u | count;
    const float value = blur.Method == MotionBlurMethod::Fast ? float(count) : 100.f;
    return archive(blur.Shutter, encoded, value);
}
template<typename Archive> constexpr auto serialize(Archive &archive, MotionBlur &blur) {
    if constexpr (Archive::kind() != decltype(Archive::kind())::in) return serialize(archive, std::as_const(blur));
    else {
        uint8_t encoded{};
        float value{};
        const auto result = archive(blur.Shutter, encoded, value);
        blur.Method = encoded > 0x80u ? MotionBlurMethod::FullSampling : MotionBlurMethod::Fast;
        blur.Steps = encoded == 0x80u ? uint8_t(value >= 1.f && value <= 64.f ? value : 16.f) :
            encoded > 0x80u           ? std::clamp<uint8_t>(encoded & 0x7fu, 1, 64) :
                                        16u;
        return result;
    }
}

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
