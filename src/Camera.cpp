#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>

glm::mat4 Camera::GetViewMatrix() const { return glm::lookAt(Position, Target, Up); }
glm::mat4 Camera::GetProjectionMatrix(float aspect_ratio) const { return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip); }
float Camera::GetDistance() const { return glm::distance(Position, Target); }

void Camera::SetPositionFromView(const glm::mat4 &view) {
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
