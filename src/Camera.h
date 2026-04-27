#pragma once

#include "numeric/vec2.h"

#include <glm/common.hpp>
#include <glm/trigonometric.hpp>
#include <optional>
#include <variant>

inline constexpr float DefaultAspectRatio{16.f / 9.f};
inline constexpr float DefaultPerspectiveNearClip{0.1f}, DefaultPerspectiveFarClip{1000.f};
inline constexpr float MinNearClip{0.01f}, MaxFarClip{DefaultPerspectiveFarClip}, MinNearFarDelta{MinNearClip};
inline constexpr float DefaultPerspectiveFieldOfViewRad{glm::radians(60.f)};

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

inline Perspective DefaultPerspectiveCamera() {
    return {.FieldOfViewRad = DefaultPerspectiveFieldOfViewRad, .FarClip = DefaultPerspectiveFarClip, .NearClip = DefaultPerspectiveNearClip};
}

inline float AspectRatio(const Camera &camera) {
    if (const auto *persp = std::get_if<Perspective>(&camera)) return persp->AspectRatio.value_or(DefaultAspectRatio);
    const auto &mag = std::get<Orthographic>(camera).Mag;
    return mag.x / mag.y;
}

inline Perspective PerspectiveFromOrthographic(const Orthographic &orthographic, std::optional<float> distance = {}) {
    const float field_of_view_rad = distance ? glm::clamp(2.f * std::atan(orthographic.Mag.y / *distance), glm::radians(1.f), glm::radians(179.f)) : glm::radians(60.f);
    return {.FieldOfViewRad = field_of_view_rad, .FarClip = orthographic.FarClip, .NearClip = orthographic.NearClip};
}

inline Orthographic OrthographicFromPerspective(const Perspective &perspective, std::optional<float> distance = {}, std::optional<float> aspect_ratio = {}) {
    vec2 mag{1.f};
    if (distance && aspect_ratio) {
        mag.y = *distance * std::tan(perspective.FieldOfViewRad * 0.5f);
        mag.x = mag.y * *aspect_ratio;
    }
    return {
        .Mag = mag,
        .FarClip = perspective.FarClip.value_or(std::max(perspective.NearClip + MinNearFarDelta, DefaultPerspectiveFarClip)),
        .NearClip = perspective.NearClip,
    };
}

inline float LookThroughFrameRatio(float camera_aspect, float viewport_aspect, float pad_ratio = 0.9f) {
    return camera_aspect > viewport_aspect ? viewport_aspect * pad_ratio / camera_aspect : pad_ratio;
}

inline Camera WidenForLookThrough(const Camera &camera, float viewport_aspect, float pad_ratio = 0.9f) {
    const float zoom = 1.f / LookThroughFrameRatio(AspectRatio(camera), viewport_aspect, pad_ratio);
    return std::visit(
        [zoom](const auto &projection) -> Camera {
            using Projection = std::decay_t<decltype(projection)>;
            if constexpr (std::is_same_v<Projection, Perspective>) {
                auto widened = projection;
                widened.FieldOfViewRad = std::min(2.f * std::atan(std::tan(projection.FieldOfViewRad * 0.5f) * zoom), glm::radians(179.f));
                return widened;
            } else {
                auto widened = projection;
                widened.Mag *= zoom;
                return widened;
            }
        },
        camera
    );
}

inline float ScreenPixelScale(const Camera &camera, float viewport_height) {
    return std::visit(
        [viewport_height](const auto &projection) -> float {
            using Projection = std::decay_t<decltype(projection)>;
            if constexpr (std::is_same_v<Projection, Perspective>) return 2.f * std::tan(projection.FieldOfViewRad * 0.5f) / viewport_height;
            else return -(2.f * projection.Mag.y / viewport_height);
        },
        camera
    );
}
