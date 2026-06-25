#include "viewport/ViewCamera.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr vec3 WorldUp{0, 1, 0};
constexpr float MinDistance{0.001f};
constexpr uint32_t DurationFrames{12}; // ~200ms at 60fps.

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

quat ViewCamera::OrientationFromAway(vec3 away) {
    away = glm::normalize(away);
    auto right = glm::cross(WorldUp, away);
    const float right_len = glm::length(right);
    // Near the poles (away ~ ±WorldUp) the horizontal axis is ambiguous; pick a default azimuth.
    right = right_len > 1e-4f ? right / right_len : vec3{1, 0, 0};
    return glm::quat_cast(mat3{right, glm::cross(away, right), away});
}

mat4 ViewCamera::View() const { return glm::lookAt(Position(), Target, Up()); }
mat4 ViewCamera::Projection(float aspect_ratio) const {
    if (const auto *perspective = std::get_if<Perspective>(&Data)) {
        if (perspective->FarClip) return glm::perspectiveRH_ZO(perspective->FieldOfViewRad, aspect_ratio, perspective->NearClip, *perspective->FarClip);
        return glm::infinitePerspectiveRH_ZO(perspective->FieldOfViewRad, aspect_ratio, perspective->NearClip);
    }

    const auto &orthographic = std::get<Orthographic>(Data);
    const vec2 mag{orthographic.Mag.y * aspect_ratio, orthographic.Mag.y};
    return glm::orthoRH_ZO(-mag.x, mag.x, -mag.y, mag.y, orthographic.NearClip, orthographic.FarClip);
}
mat3 ViewCamera::Basis() const {
    const auto m = glm::mat3_cast(Orientation); // {Right, Up, Away}
    return {m[0], m[1], -m[2]}; // {Right, Up, -Forward}
}

void ViewCamera::ApplyDistance(float new_distance) {
    if (auto *orthographic = std::get_if<Orthographic>(&Data)) orthographic->Mag *= new_distance / Distance;
    Distance = new_distance;
}

void ViewCamera::RotateBy(vec2 delta) {
    if (delta == vec2{0}) return;
    Anim.reset();
    // Turntable: yaw about world up, pitch about the camera's local right axis
    Orientation = glm::normalize(glm::angleAxis(-delta.x, WorldUp) * Orientation * glm::angleAxis(-delta.y, vec3{1, 0, 0}));
}

void ViewCamera::ZoomBy(float factor) {
    if (factor == 1.f) return;
    Anim.reset();
    ApplyDistance(std::max(Distance * factor, MinDistance));
}

void ViewCamera::SetTargetDirection(vec3 away) { AnimateTo(Target, OrientationFromAway(away), Distance); }

void ViewCamera::AnimateTo(vec3 target, quat orientation, float distance) {
    distance = std::max(distance, MinDistance);
    orientation = glm::normalize(orientation);
    // Take the shorter of the two equivalent quaternions so the slerp follows the shortest arc.
    if (glm::dot(Orientation, orientation) < 0.f) orientation = -orientation;
    if (target == Target && distance == Distance && orientation == Orientation) {
        Anim.reset();
        return;
    }
    Anim = Animation{
        .SrcTarget = Target,
        .DstTarget = target,
        .SrcDistance = Distance,
        .DstDistance = distance,
        .SrcOrientation = Orientation,
        .DstOrientation = orientation,
        .Frame = 0,
    };
}

void ViewCamera::AnimateToLookThrough(vec3 camera_position, quat orientation, float distance) {
    orientation = glm::normalize(orientation);
    distance = std::max(distance, MinDistance);
    // Position() == Target + Distance * Forward(), with Forward() == orientation * +Z at the destination.
    AnimateTo(camera_position - orientation * vec3{0, 0, 1} * distance, orientation, distance);
}

bool ViewCamera::Tick() {
    if (!Anim) return false;
    ++Anim->Frame;
    const float t = std::min(float(Anim->Frame) / float(DurationFrames), 1.f);
    const float k = Smoothstep(t);
    Target = glm::mix(Anim->SrcTarget, Anim->DstTarget, k);
    ApplyDistance(glm::mix(Anim->SrcDistance, Anim->DstDistance, k));
    Orientation = glm::slerp(Anim->SrcOrientation, Anim->DstOrientation, k);
    if (Anim->Frame >= DurationFrames) Anim.reset();
    return true;
}
