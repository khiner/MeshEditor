#pragma once

#include "numeric/vec2.h"

#include <optional>
#include <variant>

inline constexpr float DefaultAspectRatio{16.f / 9.f};
inline constexpr float DefaultPerspectiveNearClip{0.1f}, DefaultPerspectiveFarClip{1000.f};
inline constexpr float MinNearClip{0.01f}, MaxFarClip{DefaultPerspectiveFarClip}, MinNearFarDelta{MinNearClip};

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

using Camera = std::variant<Perspective, Orthographic>;
