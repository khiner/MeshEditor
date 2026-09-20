#pragma once

#include "Field.h"
#include "numeric/vec2.h"

#include <cmath>
#include <optional>
#include <variant>

using numeric::vec2;

inline constexpr float DefaultAspectRatio{16.f / 9.f};
inline constexpr float DefaultPerspectiveNearClip{0.1f}, DefaultPerspectiveFarClip{1000.f};
inline constexpr float MinNearClip{0.01f}, MaxFarClip{DefaultPerspectiveFarClip}, MinNearFarDelta{MinNearClip};

// A camera object holds exactly one of these two components.
struct Orthographic {
    vec2 Mag; // World-space half-extents of the view volume.
    float FarClip, NearClip;

    bool operator==(const Orthographic &) const = default;
};

struct Perspective {
    float FieldOfViewRad;
    float FarClip; // Infinity selects an infinite projection matrix.
    float NearClip;
    float AspectRatio{0}; // 0 uses the viewport aspect.

    bool HasFarClip() const { return std::isfinite(FarClip); }
    bool HasAspectRatio() const { return AspectRatio > 0; }
    bool operator==(const Perspective &) const = default;
};

// The view camera's lens, and the value a lens conversion produces.
using CameraLens = std::variant<Perspective, Orthographic>;

inline constexpr float MinFieldOfViewRad{0.0174533f}, MaxFieldOfViewRad{3.12414f}; // 1 and 179 degrees.
template<> inline constexpr FieldSpec Spec<Perspective, "FieldOfViewRad">{.Min = MinFieldOfViewRad, .Max = MaxFieldOfViewRad, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<Perspective, "NearClip">{.Min = MinNearClip, .Max = MaxFarClip};
template<> inline constexpr FieldSpec Spec<Perspective, "FarClip">{.Min = MinNearClip, .Max = MaxFarClip};
template<> inline constexpr FieldSpec Spec<Perspective, "AspectRatio">{.Min = 0.1, .Max = 5};
template<> inline constexpr FieldSpec Spec<Orthographic, "Mag">{.Min = 0.01, .Max = 100};
template<> inline constexpr FieldSpec Spec<Orthographic, "NearClip">{.Min = MinNearClip, .Max = MaxFarClip};
template<> inline constexpr FieldSpec Spec<Orthographic, "FarClip">{.Min = MinNearClip, .Max = MaxFarClip};
