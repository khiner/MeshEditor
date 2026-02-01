#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>

// This is specifically designed to be the view camera, not a free camera.
struct Camera {
    Camera(vec3 position, vec3 target, float field_of_view_rad, float near_clip, float far_clip)
        : Target(target), Distance(glm::length(position - target)), FieldOfViewRad(field_of_view_rad), NearClip(near_clip), FarClip(far_clip) {
        const auto direction = glm::normalize(position - target);
        YawPitch = {atan2(direction.z, direction.x), asin(direction.y)};
    }
    ~Camera() = default;

    vec3 Target;
    float Distance;
    vec2 YawPitch; // Ranges ([0, 2π], [-π, π]) If pitch is in (wrapped) range [π/2, 3π/2], camera is flipped
    float FieldOfViewRad;
    float NearClip, FarClip;

    vec3 Forward() const;
    mat3 Basis() const; // Right, Up, -Forward
    ray Ray() const { return {Position(), Forward()}; }
    mat4 View() const;
    mat4 Projection(float aspect_ratio) const;
    vec3 Position() const { return Target + Distance * Forward(); }
    ray PixelToWorldRay(vec2 mouse_px, vec2 viewport_pos, vec2 viewport_size) const;

    bool IsAligned(vec3 direction) const;
    bool IsInFront(vec3) const;

    void SetTargetDistance(float);
    void SetTargetYawPitch(vec2);
    void SetTargetDirection(vec3);

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