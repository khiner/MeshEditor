#include "Camera.h"

#include "World.h"

#include <glm/gtc/matrix_transform.hpp>

using glm::vec2, glm::vec3, glm::vec4, glm::mat4;

mat4 Camera::GetViewMatrix() const { return glm::lookAt(Position, Target, World::Up); }
mat4 Camera::GetProjectionMatrix(float aspect_ratio) const { return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip); }
mat4 Camera::GetInvViewProjectionMatrix(float aspect_ratio) const { return glm::inverse(GetProjectionMatrix(aspect_ratio) * GetViewMatrix()); }
float Camera::GetDistance() const { return glm::distance(Position, Target); }

Ray Camera::ClipPosToWorldRay(const vec2 &pos_clip, float aspect_ratio) const {
    const mat4 inv_vp = GetInvViewProjectionMatrix(aspect_ratio);

    vec4 near_point = inv_vp * vec4(pos_clip.x, pos_clip.y, -1.0, 1.0);
    near_point /= near_point.w; // Perspective divide.

    vec4 far_point = inv_vp * vec4(pos_clip.x, pos_clip.y, 1.0, 1.0);
    far_point /= far_point.w; // Perspective divide.

    return {near_point, glm::normalize(far_point - near_point)};
}

void Camera::SetPositionFromView(const mat4 &view) {
    Position = glm::inverse(view)[3];
    IsMoving = false;
}

void Camera::SetTargetDistance(float distance) {
    TargetDistance = distance;
    IsMoving = true;
}

bool Camera::Tick() {
    if (!IsMoving) return false;

    const auto distance = GetDistance();
    if (abs(distance - TargetDistance) < 0.001) {
        IsMoving = false;
        SetDistance(TargetDistance);
    } else {
        SetDistance(glm::mix(distance, TargetDistance, TickSpeed));
    }
    return true;
}

void Camera::StopMoving() {
    IsMoving = false;
    TargetDistance = GetDistance();
}

void Camera::SetDistance(float distance) {
    Position = Target + glm::normalize(Position - Target) * distance;
}
