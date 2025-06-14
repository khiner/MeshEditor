#include "Camera.h"
#include "numeric/vec4.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr bool IsClose(const vec3 &dir, const vec3 &up) { return glm::abs(glm::dot(dir, up)) > 0.999f; }
constexpr float WrapYaw(float angle) { return glm::mod(angle, glm::two_pi<float>()); }
constexpr float WrapPitch(float angle) { return glm::atan(glm::sin(angle), glm::cos(angle)); }
constexpr float LengthSq(vec2 v) { return glm::dot(v, v); }
} // namespace

bool Camera::IsAligned(vec3 direction) const { return IsClose(Forward(), direction); }

vec3 Camera::YAxis() const {
    const bool is_flipped = Pitch > glm::half_pi<float>() || Pitch < -glm::half_pi<float>();
    return {0, (is_flipped ? -1.f : 1.f), 0};
}

mat4 Camera::GetView() const { return glm::lookAt(Target + Distance * Forward(), Target, YAxis()); }
mat4 Camera::GetProjection(float aspect_ratio) const {
    return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip);
}
ray Camera::ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const {
    const auto vp_inv = glm::inverse(GetProjection(aspect_ratio) * GetView());
    auto near_point = vp_inv * vec4{pos_clip.x, pos_clip.y, -1, 1};
    near_point /= near_point.w;
    auto far_point = vp_inv * vec4{pos_clip.x, pos_clip.y, 1, 1};
    far_point /= far_point.w;
    return {near_point, glm::normalize(far_point - near_point)};
}

vec3 Camera::Forward() const { return {glm::cos(Yaw) * glm::cos(Pitch), glm::sin(Pitch), glm::sin(Yaw) * glm::cos(Pitch)}; }
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
    TargetYawPitch = yaw_pitch;
}
void Camera::AddYawPitch(vec2 yaw_pitch_delta) {
    _SetYawPitch(vec2{Yaw, Pitch} + yaw_pitch_delta);
    Changed = true;
    StopMoving();
}

bool Camera::Tick() {
    if (Changed) {
        Changed = false;
        return true;
    }
    if (YawPitchVelocity != vec2{0}) {
        _SetYawPitch(vec2{Yaw, Pitch} + YawPitchVelocity);
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
        const vec2 current{Yaw, Pitch};
        const auto delta = *TargetYawPitch - current;
        if (LengthSq(delta) < 0.0001) {
            _SetYawPitch(*TargetYawPitch);
            TargetYawPitch.reset();
        } else {
            _SetYawPitch(current + glm::mix(vec2{0}, delta, TickSpeed));
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

void Camera::_SetYawPitch(vec2 yaw_pitch) {
    Yaw = WrapYaw(yaw_pitch.x);
    Pitch = WrapPitch(yaw_pitch.y);
}
