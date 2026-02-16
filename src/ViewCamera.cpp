#include "ViewCamera.h"
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

inline constexpr float MinDistance{0.001f};
} // namespace

bool ViewCamera::IsAligned(vec3 direction) const { return glm::dot(Forward(), glm::normalize(direction)) > 0.999f; }
bool ViewCamera::IsInFront(vec3 p) const { return glm::dot(p - Position(), -Forward()) > NearClip(); }

float ViewCamera::NearClip() const {
    if (const auto *perspective = std::get_if<Perspective>(&Data)) return perspective->NearClip;
    return std::get<Orthographic>(Data).NearClip;
}

float ViewCamera::FarClip() const {
    if (const auto *perspective = std::get_if<Perspective>(&Data)) return perspective->FarClip.value_or(MaxFarClip);
    return std::get<Orthographic>(Data).FarClip;
}

ray ViewCamera::PixelToWorldRay(vec2 mouse_px, rect viewport) const {
    const auto rel = (mouse_px - viewport.pos) / viewport.size;
    const auto ndc = vec2{rel.x, 1.f - rel.y} * 2.f - 1.f;
    if (const auto *perspective = std::get_if<Perspective>(&Data)) {
        const auto aspect = viewport.size.x / viewport.size.y;
        const auto t = std::tan(perspective->FieldOfViewRad * 0.5f);
        // View-space direction with +Z along the camera view dir, rotated into world-space
        return {Position(), Basis() * glm::normalize(vec3{ndc.x * aspect * t, ndc.y * t, 1.f})};
    }

    const auto &orthographic = std::get<Orthographic>(Data);
    const auto basis = Basis();
    const auto view_dir = -Forward();
    const auto aspect = viewport.size.x / viewport.size.y;
    const vec2 mag{orthographic.Mag.y * aspect, orthographic.Mag.y};
    return {Position() + view_dir * NearClip() + basis[0] * (ndc.x * mag.x) + basis[1] * (ndc.y * mag.y), view_dir};
}

vec3 ViewCamera::YAxis() const {
    const bool is_flipped = YawPitch.y > glm::half_pi<float>() || YawPitch.y < -glm::half_pi<float>();
    return {0, (is_flipped ? -1.f : 1.f), 0};
}

mat4 ViewCamera::View() const { return glm::lookAt(Position(), Target, YAxis()); }
mat4 ViewCamera::Projection(float aspect_ratio) const {
    if (const auto *perspective = std::get_if<Perspective>(&Data)) {
        if (perspective->FarClip) return glm::perspectiveRH_ZO(perspective->FieldOfViewRad, aspect_ratio, perspective->NearClip, *perspective->FarClip);
        return glm::infinitePerspectiveRH_ZO(perspective->FieldOfViewRad, aspect_ratio, perspective->NearClip);
    }

    const auto &orthographic = std::get<Orthographic>(Data);
    const vec2 mag{orthographic.Mag.y * aspect_ratio, orthographic.Mag.y};
    return glm::orthoRH_ZO(-mag.x, mag.x, -mag.y, mag.y, orthographic.NearClip, orthographic.FarClip);
}
vec3 ViewCamera::Forward() const { return {glm::cos(YawPitch.x) * glm::cos(YawPitch.y), glm::sin(YawPitch.y), glm::sin(YawPitch.x) * glm::cos(YawPitch.y)}; }
mat3 ViewCamera::Basis() const {
    const auto forward = Forward();
    const auto right = glm::normalize(glm::cross(YAxis(), forward));
    return {right, glm::cross(forward, right), -forward};
}

void ViewCamera::SetTargetDirection(vec3 direction) {
    SetTargetYawPitch({WrapYaw(atan2(direction.z, direction.x)), WrapPitch(asin(direction.y))});
}
void ViewCamera::SetTargetDistance(float distance) {
    StopMoving();
    GoalDistance = std::max(distance, MinDistance);
}
void ViewCamera::SetTargetYawPitch(vec2 yaw_pitch) {
    StopMoving();
    GoalYawPitch = {WrapYaw(yaw_pitch.x), WrapPitch(yaw_pitch.y)};
}

void ViewCamera::SetData(const CameraData &camera_data) {
    Data = camera_data;
    StopMoving();
}

void ViewCamera::AnimateTo(vec3 target, vec2 yaw_pitch, float distance) {
    StopMoving();
    GoalTarget = target;
    GoalDistance = std::max(distance, MinDistance);
    GoalYawPitch = {WrapYaw(yaw_pitch.x), WrapPitch(yaw_pitch.y)};
}

bool ViewCamera::Tick() {
    if (Changed) {
        Changed = false;
        return true;
    }
    bool moved = false;
    if (YawPitchVelocity != vec2{0}) {
        _SetYawPitch(YawPitch + YawPitchVelocity);
        YawPitchVelocity *= 0.9f;
        if (LengthSq(YawPitchVelocity) < 0.00001) YawPitchVelocity = {};
        moved = true;
    }
    if (GoalTarget) {
        if (glm::length(*GoalTarget - Target) < 0.0001f) {
            Target = *GoalTarget;
            GoalTarget.reset();
        } else {
            Target = glm::mix(Target, *GoalTarget, TickSpeed);
        }
        moved = true;
    }
    if (GoalDistance) {
        const auto old_distance = Distance;
        if (std::abs(old_distance - *GoalDistance) < 0.0001) {
            Distance = *GoalDistance;
            GoalDistance.reset();
        } else {
            Distance = glm::mix(old_distance, *GoalDistance, TickSpeed);
        }
        if (auto *orthographic = std::get_if<Orthographic>(&Data)) {
            orthographic->Mag *= Distance / old_distance;
        }
        moved = true;
    }
    if (GoalYawPitch) {
        const vec2 delta{ShortestAngleDelta(YawPitch.x, GoalYawPitch->x), ShortestAngleDelta(YawPitch.y, GoalYawPitch->y)};
        if (LengthSq(delta) < 0.0001f) {
            _SetYawPitch(*GoalYawPitch);
            GoalYawPitch.reset();
        } else {
            _SetYawPitch(YawPitch + delta * TickSpeed); // minimal path step
        }
        moved = true;
    }
    return moved;
}

bool ViewCamera::IsAnimating() const { return GoalTarget || GoalDistance || GoalYawPitch || YawPitchVelocity != vec2{0}; }

void ViewCamera::StopMoving() {
    GoalTarget.reset();
    GoalDistance.reset();
    GoalYawPitch.reset();
    YawPitchVelocity = {};
}

void ViewCamera::_SetYawPitch(vec2 yaw_pitch) { YawPitch = {WrapYaw(yaw_pitch.x), WrapPitch(yaw_pitch.y)}; }
