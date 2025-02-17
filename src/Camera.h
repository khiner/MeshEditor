#pragma once

#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/vec2.h"

#include <optional>

struct Camera {
    Camera(vec3 position, vec3 target, float field_of_view, float near_clip, float far_clip)
        : Target(target), Distance(glm::length(position - target)), FieldOfView(field_of_view), NearClip(near_clip), FarClip(far_clip) {
        Orientation = glm::quatLookAt(glm::normalize(position - target), vec3{0, 1, 0});
    }
    ~Camera() = default;

    vec3 Target;
    float Distance;
    quat Orientation;
    float FieldOfView;
    float NearClip, FarClip;

    mat4 GetView() const;
    mat4 GetProjection(float aspect_ratio) const;
    ray ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const;
    void SetTargetDistance(float distance) { TargetDistance = distance; }
    void SetTargetDirection(vec3 direction) { TargetDirection = glm::normalize(direction); }
    void OrbitDelta(vec2 angles_delta);
    bool Tick();
    void StopMoving();

    bool IsAligned(vec3 direction) const;

private:
    std::optional<float> TargetDistance{};
    std::optional<vec3> TargetDirection{};
    bool Changed{false};
    float TickSpeed{0.25};

    void SetDistance(float distance);
};