#include "Camera.h"
#include "numeric/vec4.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float WrapYaw(float angle) { return glm::mod(angle, glm::two_pi<float>()); }
constexpr float WrapPitch(float angle) { return glm::atan(glm::sin(angle), glm::cos(angle)); }
constexpr float LengthSq(vec2 v) { return glm::dot(v, v); }
constexpr float ShortestAngleDelta(float from, float to) {
    const float d = to - from;
    return std::atan2(std::sin(d), std::cos(d)); // in (-pi, pi]
}
} // namespace

bool Camera::IsAligned(vec3 direction) const { return glm::dot(Forward(), glm::normalize(direction)) > 0.999f; }
bool Camera::IsInFront(vec3 p) const { return glm::dot(p - Position(), -Forward()) > NearClip; }

ray Camera::NdcToWorldRay(vec2 ndc, float aspect_ratio) const {
    const float t = std::tan(FieldOfViewRad * 0.5f);
    // View-space direction with +Z along the camera view dir, rotated into world-space
    return {Position(), Basis() * glm::normalize(vec3{ndc.x * aspect_ratio * t, ndc.y * t, 1.f})};
}

vec3 Camera::YAxis() const {
    const bool is_flipped = YawPitch.y > glm::half_pi<float>() || YawPitch.y < -glm::half_pi<float>();
    return {0, (is_flipped ? -1.f : 1.f), 0};
}

mat4 Camera::View() const { return glm::lookAt(Position(), Target, YAxis()); }
mat4 Camera::Projection(float aspect_ratio) const {
    return glm::perspective(FieldOfViewRad, aspect_ratio, NearClip, FarClip);
}
vec3 Camera::Forward() const { return {glm::cos(YawPitch.x) * glm::cos(YawPitch.y), glm::sin(YawPitch.y), glm::sin(YawPitch.x) * glm::cos(YawPitch.y)}; }
mat3 Camera::Basis() const {
    const auto forward = Forward();
    const auto right = glm::normalize(glm::cross(YAxis(), forward));
    return {right, glm::cross(forward, right), -forward};
}

void Camera::SetTargetDirection(vec3 direction) {
    SetTargetYawPitch({WrapYaw(atan2(direction.z, direction.x)), WrapPitch(asin(direction.y))});
}
void Camera::SetTargetDistance(float distance) {
    StopMoving();
    TargetDistance = distance;
}
void Camera::SetTargetYawPitch(vec2 yaw_pitch) {
    StopMoving();
    TargetYawPitch = {WrapYaw(yaw_pitch.x), WrapPitch(yaw_pitch.y)};
}

bool Camera::Tick() {
    if (Changed) {
        Changed = false;
        return true;
    }
    if (YawPitchVelocity != vec2{0}) {
        _SetYawPitch(YawPitch + YawPitchVelocity);
        YawPitchVelocity *= 0.9f;
        if (LengthSq(YawPitchVelocity) < 0.00001) YawPitchVelocity = {};
        return true;
    }
    if (TargetDistance) {
        const auto distance = Distance;
        if (std::abs(distance - *TargetDistance) < 0.0001) {
            Distance = *TargetDistance;
            TargetDistance.reset();
        } else {
            Distance = glm::mix(distance, *TargetDistance, TickSpeed);
        }
        return true;
    }
    if (TargetYawPitch) {
        const vec2 delta{
            ShortestAngleDelta(YawPitch.x, TargetYawPitch->x),
            ShortestAngleDelta(YawPitch.y, TargetYawPitch->y)
        };
        if (LengthSq(delta) < 0.0001f) {
            _SetYawPitch(*TargetYawPitch);
            TargetYawPitch.reset();
        } else {
            _SetYawPitch(YawPitch + delta * TickSpeed); // minimal path step
        }
        return true;
    }
    return false;
}

void Camera::StopMoving() {
    TargetDistance.reset();
    TargetYawPitch.reset();
    YawPitchVelocity = {};
}

void Camera::_SetYawPitch(vec2 yaw_pitch) { YawPitch = {WrapYaw(yaw_pitch.x), WrapPitch(yaw_pitch.y)}; }
