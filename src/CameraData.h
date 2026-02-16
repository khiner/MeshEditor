#pragma once

#include <cstdint>
#include <optional>
#include <variant>

struct Orthographic {
    float XMag, YMag;
    float FarClip, NearClip;
};

struct Perspective {
    float FieldOfViewRad;
    std::optional<float> FarClip; // If omitted, use an infinite projection matrix.
    float NearClip;
    std::optional<float> AspectRatio{};
};

using CameraData = std::variant<Perspective, Orthographic>;
