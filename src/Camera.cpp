#include "Camera.h"
#include "numeric/vec4.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
const vec3 YAxis{0, 1, 0}, XAxis{1, 0, 0};
constexpr bool Close(const vec3 &dir, const vec3 &up) { return glm::abs(glm::dot(dir, up)) > 0.9999f; }

quat GetOrbitDelta(const quat &orientation, vec2 angles_delta) {
    // const bool is_pole = Close(orientation * vec3{0, 0, 1}, YAxis);
    // todo Need to handle poles differently to avoid skewing camera.
    const auto yaw_rotation = glm::angleAxis(angles_delta.x, YAxis);
    const auto pitch_rotation = glm::angleAxis(angles_delta.y, orientation * XAxis);
    return glm::normalize(pitch_rotation * yaw_rotation * orientation);
}
} // namespace

bool Camera::IsAligned(vec3 direction) const {
    return Close(glm::normalize(direction), Orientation * vec3{0, 0, 1});
}
mat4 Camera::GetView() const {
    const auto position = Target - (Orientation * vec3{0, 0, Distance});
    return glm::lookAt(position, Target, Orientation * vec3{0, 1, 0});
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

void Camera::OrbitDelta(vec2 angles_delta) {
    Orientation = GetOrbitDelta(Orientation, angles_delta);
    Changed = true;
    TargetDirection.reset();
    TargetDistance.reset();
}

bool Camera::Tick() {
    if (Changed) {
        Changed = false;
        return true;
    }
    if (!TargetDistance && !TargetDirection) return false;

    if (TargetDistance) {
        const auto distance = Distance;
        if (std::abs(distance - *TargetDistance) < 0.0001) {
            SetDistance(*TargetDistance);
            TargetDistance.reset();
        } else {
            SetDistance(glm::mix(distance, *TargetDistance, TickSpeed));
        }
    }
    if (TargetDirection) {
        const auto current_direction = glm::normalize(Target - (Orientation * vec3{0, 0, Distance}));
        const vec2 current_angles{
            atan2(current_direction.x, current_direction.z),
            asin(glm::clamp(current_direction.y, -1.f, 1.f))
        };
        const vec2 target_angles{
            atan2(TargetDirection->x, TargetDirection->z),
            asin(glm::clamp(TargetDirection->y, -1.f, 1.f))
        };
        // If the target or current dir is a pole, keep the current yaw.
        const bool target_is_pole = Close(*TargetDirection, YAxis), current_is_pole = Close(current_direction, YAxis);
        const auto delta_angles = target_is_pole || current_is_pole ? vec2{0, target_angles.y - current_angles.y} : target_angles - current_angles;
        if (glm::length(delta_angles) < 0.01) {
            TargetDirection.reset();
        } else {
            // Pass the deltas to OrbitDelta to apply smooth rotation
            Orientation = GetOrbitDelta(Orientation, glm::mix(vec2{0}, delta_angles, TickSpeed));
        }
    }
    return true;
}

void Camera::StopMoving() {
    TargetDistance.reset();
    TargetDirection.reset();
}

void Camera::SetDistance(float distance) {
    Distance = distance;
}
