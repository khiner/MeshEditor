#include "Camera.h"
#include "numeric/vec4.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
const vec3 YAxis{0, 1, 0};
constexpr bool Close(const vec3 &dir, const vec3 &up) { return glm::abs(glm::dot(dir, up)) > 0.9999f; }
// Wrap angle to [-pi, pi]
constexpr float WrapYaw(float angle) { return glm::mod(angle, glm::two_pi<float>()); }
constexpr float WrapPitch(float angle) { return glm::atan(glm::sin(angle), glm::cos(angle)); }
constexpr float LengthSq(vec2 v) { return glm::dot(v, v); }
} // namespace

bool Camera::IsAligned(vec3 direction) const {
    const auto current_dir = glm::normalize(vec3{glm::cos(Yaw) * glm::cos(Pitch), glm::sin(Pitch), glm::sin(Yaw) * glm::cos(Pitch)});
    return Close(current_dir, direction);
}
mat4 Camera::GetView() const {
    const bool is_flipped = Pitch > glm::half_pi<float>() || Pitch < -glm::half_pi<float>();
    return glm::lookAt(Target + Distance * vec3{glm::cos(Yaw) * glm::cos(Pitch), glm::sin(Pitch), glm::sin(Yaw) * glm::cos(Pitch)}, Target, YAxis * (is_flipped ? -1.f : 1.f));
}

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

void Camera::SetTargetDirection(vec3 direction) {
    SetTargetYawPitch({WrapYaw(atan2(direction.z, direction.x)), WrapPitch(asin(direction.y))});
    YawPitchVelocity = {};
}
void Camera::SetTargetDistance(float distance) {
    TargetDistance = distance;
    YawPitchVelocity = {};
}
void Camera::SetTargetYawPitch(vec2 yaw_pitch) {
    TargetYawPitch = yaw_pitch;
    YawPitchVelocity = {};
}
void Camera::AddYawPitch(vec2 yaw_pitch_delta) {
    _SetYawPitch(vec2{Yaw, Pitch} + yaw_pitch_delta);
    Changed = true;
    TargetYawPitch.reset();
    TargetDistance.reset();
    YawPitchVelocity = {};
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
    if (!TargetDistance && !TargetYawPitch) return false;

    if (TargetDistance) {
        const auto distance = Distance;
        if (std::abs(distance - *TargetDistance) < 0.0001) {
            Distance = *TargetDistance;
            TargetDistance.reset();
        } else {
            Distance = glm::mix(distance, *TargetDistance, TickSpeed);
        }
    }
    if (TargetYawPitch) {
        const vec2 current{Yaw, Pitch};
        const auto delta = *TargetYawPitch - current;
        if (LengthSq(delta) < 0.0001) {
            TargetYawPitch.reset();
        } else {
            _SetYawPitch(current + glm::mix(vec2{0}, delta, TickSpeed));
        }
    }
    return true;
}

void Camera::StopMoving() {
    TargetDistance.reset();
    TargetYawPitch.reset();
}

void Camera::_SetYawPitch(vec2 yaw_pitch) {
    Yaw = WrapYaw(yaw_pitch.x);
    Pitch = WrapPitch(yaw_pitch.y);
}
