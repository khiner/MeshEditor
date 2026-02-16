#pragma once

#include "numeric/vec2.h"

#include <optional>
#include <variant>

inline constexpr float DefaultAspectRatio{16.f / 9.f};
inline constexpr float MinNearFarDelta{0.001f}, MaxFarClip{100.f};

struct Orthographic {
    vec2 Mag; // x/y half-extents of the view volume in world units
    float FarClip, NearClip;
};

struct Perspective {
    float FieldOfViewRad;
    std::optional<float> FarClip; // If omitted, use an infinite projection matrix
    float NearClip;
    std::optional<float> AspectRatio{};
};

using CameraData = std::variant<Perspective, Orthographic>;

inline float AspectRatio(const CameraData &cd) {
    if (const auto *persp = std::get_if<Perspective>(&cd)) return persp->AspectRatio.value_or(DefaultAspectRatio);
    const auto &mag = std::get<Orthographic>(cd).Mag;
    return mag.x / mag.y;
}
