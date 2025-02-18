#include "Camera.h"
#include "numeric/vec4.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
const vec3 YAxis{0, 1, 0};
constexpr bool Close(const vec3 &dir, const vec3 &up) { return glm::abs(glm::dot(dir, up)) > 0.9999f; }
// Wrap angle to [-pi, pi]
constexpr float WrapYaw(float angle) { return glm::mod(angle, glm::two_pi<float>()); }
constexpr float WrapPitch(float angle) { return glm::atan(glm::sin(angle), glm::cos(angle)); }
} // namespace

void Camera::SetTargetDirection(vec3 direction) {
    SetTargetYawPitch({WrapYaw(atan2(direction.z, direction.x)), WrapPitch(asin(direction.y))});
}
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

void Camera::SetYawPitch(vec2 yaw_pitch) {
    Yaw = WrapYaw(yaw_pitch.x);
    Pitch = WrapPitch(yaw_pitch.y);
}

void Camera::AddYawPitch(vec2 yaw_pitch_delta) {
    SetYawPitch(vec2{Yaw, Pitch} + yaw_pitch_delta);
    Changed = true;
    TargetYawPitch.reset();
    TargetDistance.reset();
}

bool Camera::Tick() {
    if (Changed) {
        Changed = false;
        return true;
    }
    if (!TargetDistance && !TargetYawPitch) return false;

    if (TargetDistance) {
        const auto distance = Distance;
        if (std::abs(distance - *TargetDistance) < 0.0001) {
            SetDistance(*TargetDistance);
            TargetDistance.reset();
        } else {
            SetDistance(glm::mix(distance, *TargetDistance, TickSpeed));
        }
    }
    if (TargetYawPitch) {
        const vec2 current{Yaw, Pitch};
        const vec2 target = *TargetYawPitch;
        // If the target or current dir is a pole, keep the current yaw.
        const bool target_is_pole = abs(abs(target.y) - glm::half_pi<float>()) < 0.01;
        const bool current_is_pole = abs(abs(current.y) - glm::half_pi<float>()) < 0.01;
        const auto delta = target_is_pole || current_is_pole ? vec2{0, target.y - current.y} : target - current;
        if (glm::length(delta) < 0.01) {
            TargetYawPitch.reset();
        } else {
            SetYawPitch(current + glm::mix(vec2{0}, delta, TickSpeed));
        }
    }
    return true;
}

void Camera::StopMoving() {
    TargetDistance.reset();
    TargetYawPitch.reset();
}

void Camera::SetDistance(float distance) {
    Distance = distance;
}
