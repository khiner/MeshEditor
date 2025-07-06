#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>

// This is specifically designed to be the view camera, not a free camera.
struct Camera {
    Camera(vec3 position, vec3 target, float field_of_view, float near_clip, float far_clip)
        : Target(target), Distance(glm::length(position - target)), FieldOfView(field_of_view), NearClip(near_clip), FarClip(far_clip) {
        const auto direction = glm::normalize(position - target);
        Yaw = atan2(direction.z, direction.x);
        Pitch = asin(direction.y);
    }
    ~Camera() = default;

    vec3 Target;
    float Distance;
    float Yaw; // Range [0, 2π]
    float Pitch; // Range [-π, π]. If in (wrapped) range [π/2, 3π/2], camera is flipped.
    float FieldOfView;
    float NearClip, FarClip;

    mat4 GetView() const;
    mat4 GetProjection(float aspect_ratio) const;
    vec3 GetPosition() const { return Target + Distance * Forward(); }

    vec3 Forward() const;
    mat3 Basis() const;

    bool IsAligned(vec3 direction) const;

    void SetTargetDistance(float);
    void SetTargetYawPitch(vec2);
    void SetTargetDirection(vec3);
    void AddYawPitch(vec2);

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