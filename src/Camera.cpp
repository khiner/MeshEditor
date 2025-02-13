#include "Camera.h"
#include "numeric/vec4.h"

#include <glm/gtc/matrix_transform.hpp>

mat4 Camera::GetView() const { return glm::lookAt(Position, Target, Up); }
mat4 Camera::GetProjection(float aspect_ratio) const { return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip); }
float Camera::GetDistance() const { return glm::distance(Position, Target); }

ray Camera::ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const {
    const mat4 inv_vp = glm::inverse(GetProjection(aspect_ratio) * GetView());
    vec4 near_point = inv_vp * vec4(pos_clip.x, pos_clip.y, -1.0, 1.0);
    near_point /= near_point.w; // Perspective divide.

    vec4 far_point = inv_vp * vec4(pos_clip.x, pos_clip.y, 1.0, 1.0);
    far_point /= far_point.w; // Perspective divide.

    return {near_point, glm::normalize(far_point - near_point)};
}

void Camera::SetPositionFromView(const mat4 &view) {
    Position = glm::inverse(view)[3];
    StopMoving();
}

// Direction vector to spherical angles (azimuth and elevation).
namespace {
constexpr vec2 DirToAngles(const vec3 &dir, const vec3 &up) {
    return {atan2(dir.z, dir.x), asin(glm::clamp(glm::dot(dir, up), -1.0f, 1.0f))};
}
} // namespace

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
            TargetDistance.reset();
            if (!TargetDirection) return false;
        } else {
            SetDistance(glm::mix(distance, *TargetDistance, TickSpeed));
        }
    }
    if (TargetDirection) {
        const auto direction = glm::normalize(Position - Target);
        if (abs(glm::dot(direction, *TargetDirection) - 1.0f) < 0.001) {
            TargetDirection.reset();
            return false;
        }
        const auto direction_next = glm::mix(direction, *TargetDirection, TickSpeed);
        OrbitDelta(DirToAngles(direction_next, Up) - DirToAngles(direction, Up));
    }
    return true;
}

void Camera::SetDistance(float distance) {
    Position = Target + glm::normalize(Position - Target) * distance;
}
