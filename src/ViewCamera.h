#pragma once

#include "CameraData.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/ray.h"
#include "numeric/rect.h"
#include "numeric/vec3.h"

#include <optional>

// The viewport navigation camera (orbit/zoom around a target point).
// Note: Aspect ratio of the provided data is ignored, as the ViewCamera follows the viewport aspect ratio.
struct ViewCamera {
    ViewCamera(vec3 position, vec3 target, CameraData data)
        : Data{data}, Target{target}, Distance{glm::length(position - target)} {
        const auto direction = glm::normalize(position - target);
        YawPitch = {atan2(direction.z, direction.x), asin(direction.y)};
    }
    ~ViewCamera() = default;

    CameraData Data;
    vec3 Target;
    float Distance;
    vec2 YawPitch; // Ranges ([0, 2π], [-π, π]) If pitch is in (wrapped) range [π/2, 3π/2], camera is flipped

    float NearClip() const;
    float FarClip() const; // Always finite (fallback when perspective far is infinite).

    vec3 Forward() const;
    mat3 Basis() const; // Right, Up, -Forward
    ray Ray() const { return {Position(), Forward()}; }
    mat4 View() const;
    mat4 Projection(float aspect_ratio) const;
    vec3 Position() const { return Target + Distance * Forward(); }
    ray PixelToWorldRay(vec2 mouse_px, rect viewport) const;

    bool IsAligned(vec3 direction) const;
    bool IsInFront(vec3) const;

    void SetTargetDistance(float);
    void SetTargetYawPitch(vec2);
    void SetTargetDirection(vec3);
    void SetData(const CameraData &);

    // Not currently used, since I need to figure out trackpad touch events.
    void SetYawPitchVelocity(vec2 vel) { YawPitchVelocity = vel; }
    void StopMoving();

    bool Tick();

private:
    std::optional<float> TargetDistance{};
    std::optional<vec2> TargetYawPitch{};
    vec2 YawPitchVelocity{};
    bool Changed{false};
    float TickSpeed{0.25};

    void _SetYawPitch(vec2);
    vec3 YAxis() const; // May be flipped depending on pitch.
};
