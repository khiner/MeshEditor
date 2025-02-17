#include "Camera.h"
#include "numeric/vec4.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {
// Direction vector to spherical angles (azimuth and elevation).
constexpr vec2 DirToAngles(const vec3 &dir, const vec3 &up) {
    return {atan2(dir.z, dir.x), asin(glm::clamp(glm::dot(dir, up), -1.f, 1.f))};
}
constexpr bool NearPole(const vec3 &dir, const vec3 &up) { return glm::abs(glm::dot(dir, up)) > 0.9999f; }
} // namespace

mat4 Camera::GetView() const {
    // Detect if the camera is near the up/down pole to avoid degenerate view matrix.
    const auto direction = glm::normalize(Position - Target);
    const auto near_pole = NearPole(direction, Up);
    const auto safe_up = near_pole ? glm::cross(direction, glm::vec3{1, 0, 0}) : Up;
    return glm::lookAt(Position, Target, safe_up);
}

mat4 Camera::GetProjection(float aspect_ratio) const { return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip); }
float Camera::GetDistance() const { return glm::distance(Position, Target); }

ray Camera::ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const {
    const auto vp_inv = glm::inverse(GetProjection(aspect_ratio) * GetView());
    auto near_point = vp_inv * vec4{pos_clip.x, pos_clip.y, -1, 1};
    near_point /= near_point.w;
    auto far_point = vp_inv * vec4{pos_clip.x, pos_clip.y, 1, 1};
    far_point /= far_point.w;
    return {near_point, glm::normalize(far_point - near_point)};
}

void Camera::OrbitDelta(vec2 angles_delta) {
    const auto dir = glm::normalize(Target - Position);
    // Convert to spherical coordinates, apply deltas, and clamp elevation.
    static constexpr float MaxElevationRad = glm::radians(89.0f);
    const vec2 angles_curr = DirToAngles(dir, Up);
    const vec2 angles{
        angles_curr.x + angles_delta.x,
        glm::clamp(angles_curr.y - angles_delta.y, -MaxElevationRad, MaxElevationRad)
    };
    // Convert spherical back to Cartesian and update position.
    const vec3 new_dir{cos(angles.y) * cos(angles.x), sin(angles.y), cos(angles.y) * sin(angles.x)};
    Position = Target - new_dir * GetDistance();
}

bool Camera::Tick() {
    if (!TargetDistance && !TargetDirection) return false;

    if (TargetDistance) {
        const auto distance = GetDistance();
        if (abs(distance - *TargetDistance) < 0.001) {
            SetDistance(*TargetDistance);
            TargetDistance.reset();
        } else {
            SetDistance(glm::mix(distance, *TargetDistance, TickSpeed));
        }
    }
    if (TargetDirection) {
        if (Immediate) {
            Position = (Target - *TargetDirection) * GetDistance();
            TargetDirection.reset();
            Immediate = false;
            return true;
        }
        const auto direction = glm::normalize(Position - Target);
        if (abs(glm::dot(direction, *TargetDirection) - 1.0f) < 0.001) {
            Position = *TargetDirection * GetDistance();
            TargetDirection.reset();
        } else {
            const auto direction_next = glm::mix(direction, *TargetDirection, TickSpeed);
            OrbitDelta(DirToAngles(direction_next, Up) - DirToAngles(direction, Up));
        }
    }
    return true;
}

void Camera::SetDistance(float distance) {
    Position = Target + glm::normalize(Position - Target) * distance;
}
