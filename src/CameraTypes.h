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

// Each camera object has exactly one of these two lens components.
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

using CameraLens = std::variant<Perspective, Orthographic>;

inline constexpr float MinFieldOfViewRad{0.0174533f}, MaxFieldOfViewRad{3.12414f}; // 1 and 179 degrees.
template<> inline constexpr field::SpecTable<Perspective> Specs<Perspective>{
    {&Perspective::FieldOfViewRad, {.Min = MinFieldOfViewRad, .Max = MaxFieldOfViewRad, .Unit = FieldUnit::Radians, .Label = "Field of view", .Help = "Vertical view angle"}},
    {&Perspective::NearClip, {.Min = MinNearClip, .Max = MaxFarClip, .Unit = FieldUnit::Meters, .Help = "Distance to the near clipping plane"}},
    {&Perspective::FarClip, {.Min = MinNearClip, .Unit = FieldUnit::Meters, .Help = "Distance to the far clipping plane. Infinity selects an infinite projection."}},
    {&Perspective::AspectRatio, {.Min = 0, .Max = 5, .Help = "Width over height of the view. Zero uses the viewport aspect ratio."}},
};
template<> inline constexpr field::SpecTable<Orthographic> Specs<Orthographic>{
    {&Orthographic::Mag, {.Min = 0.01, .Max = 100, .Unit = FieldUnit::Meters, .Help = "Half the view volume's width and height"}},
    {&Orthographic::FarClip, {.Min = MinNearClip, .Max = MaxFarClip, .Unit = FieldUnit::Meters, .Help = "Distance to the far clipping plane"}},
    {&Orthographic::NearClip, {.Min = MinNearClip, .Max = MaxFarClip, .Unit = FieldUnit::Meters, .Help = "Distance to the near clipping plane"}},
};
