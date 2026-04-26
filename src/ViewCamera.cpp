#include "ViewCamera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float WrapYaw(float angle) { return glm::mod(angle, glm::two_pi<float>()); }
constexpr float WrapPitch(float angle) { return glm::atan(glm::sin(angle), glm::cos(angle)); }
constexpr float ShortestAngleDelta(float from, float to) {
    const float d = to - from;
    return std::atan2(std::sin(d), std::cos(d)); // in (-pi, pi]
}

inline constexpr float MinDistance{0.001f};
inline constexpr uint32_t DurationFrames{12}; // ~200ms at 60fps.

constexpr float Smoothstep(float t) { return t * t * (3.f - 2.f * t); }
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

void ViewCamera::ApplyDistance(float new_distance) {
    if (auto *orthographic = std::get_if<Orthographic>(&Data)) orthographic->Mag *= new_distance / Distance;
    Distance = new_distance;
}

void ViewCamera::RotateBy(vec2 delta) {
    if (delta == vec2{0}) return;
    Anim.reset();
    YawPitch = {WrapYaw(YawPitch.x + delta.x), WrapPitch(YawPitch.y + delta.y)};
}

void ViewCamera::ZoomBy(float factor) {
    if (factor == 1.f) return;
    Anim.reset();
    ApplyDistance(std::max(Distance * factor, MinDistance));
}

void ViewCamera::SetTargetDirection(vec3 direction) {
    AnimateTo(Target, {atan2(direction.z, direction.x), asin(direction.y)}, Distance);
}

void ViewCamera::AnimateTo(vec3 target, vec2 yaw_pitch, float distance) {
    distance = std::max(distance, MinDistance);
    // Unwrap dst yaw/pitch around src so a straight lerp takes the shortest path through angle space.
    const vec2 dst_yp{
        YawPitch.x + ShortestAngleDelta(YawPitch.x, WrapYaw(yaw_pitch.x)),
        YawPitch.y + ShortestAngleDelta(YawPitch.y, WrapPitch(yaw_pitch.y)),
    };
    if (target == Target && distance == Distance && dst_yp == YawPitch) {
        Anim.reset();
        return;
    }
    Anim = Animation{
        .SrcTarget = Target,
        .DstTarget = target,
        .SrcDistance = Distance,
        .DstDistance = distance,
        .SrcYawPitch = YawPitch,
        .DstYawPitch = dst_yp,
        .Frame = 0,
    };
}

bool ViewCamera::Tick() {
    if (!Anim) return false;
    ++Anim->Frame;
    const float t = std::min(float(Anim->Frame) / float(DurationFrames), 1.f);
    const float k = Smoothstep(t);
    Target = glm::mix(Anim->SrcTarget, Anim->DstTarget, k);
    ApplyDistance(glm::mix(Anim->SrcDistance, Anim->DstDistance, k));
    YawPitch = {WrapYaw(glm::mix(Anim->SrcYawPitch.x, Anim->DstYawPitch.x, k)), WrapPitch(glm::mix(Anim->SrcYawPitch.y, Anim->DstYawPitch.y, k))};
    if (Anim->Frame >= DurationFrames) Anim.reset();
    return true;
}
