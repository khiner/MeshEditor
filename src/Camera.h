#pragma once

#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"

#include <optional>

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
    ray ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const;
    void SetTargetDistance(float distance) { TargetDistance = distance; }
    void SetTargetYawPitch(vec2 yaw_pitch) { TargetYawPitch = yaw_pitch; }
    void SetTargetDirection(vec3);
    void AddYawPitch(vec2);
    bool Tick();
    void StopMoving();

    bool IsAligned(vec3 direction) const;

private:
    std::optional<float> TargetDistance{};
    std::optional<vec2> TargetYawPitch{};
    bool Changed{false};
    float TickSpeed{0.25};

    void SetDistance(float distance);
    void SetYawPitch(vec2);
};