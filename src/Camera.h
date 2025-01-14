#pragma once

#include "Ray.h"

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include <optional>

struct Camera {
    Camera(vec3 up, vec3 position, vec3 target, float field_of_view, float near_clip, float far_clip)
        : Up(up), Position(position), Target(target), FieldOfView(field_of_view), NearClip(near_clip), FarClip(far_clip) {}
    ~Camera() = default;

    vec3 Up;
    vec3 Position;
    vec3 Target;
    float FieldOfView;
    float NearClip, FarClip;

    mat4 GetViewMatrix() const;
    mat4 GetProjectionMatrix(float aspect_ratio) const;
    mat4 GetInvViewProjectionMatrix(float aspect_ratio) const;
    float GetDistance() const;

    // Converts a position in clip space to a ray in world space.
    Ray ClipPosToWorldRay(vec2 pos_clip, float aspect_ratio) const;

    void SetPositionFromView(const mat4 &view);
    void SetTargetDistance(float distance) { TargetDistance = distance; }
    // Orbit spherically around the target until landing at the given direction.
    // Distance to target is preserved.
    void SetTargetDirection(vec3 direction) { TargetDirection = std::move(direction); }

    // Orbit spherically around the target by the given angle deltas.
    // (azimuth, elevation) / (yaw, pitch) / (x, y) / (phi, theta)
    void OrbitDelta(vec2 angles_delta);

    bool Tick(); // Move camera to target distance. Returns true if camera moved.

    void StopMoving() {
        TargetDistance.reset();
        TargetDirection.reset();
    }

private:
    std::optional<float> TargetDistance{};
    std::optional<vec3> TargetDirection{};
    float TickSpeed{0.3};

    void SetDistance(float distance);
};
