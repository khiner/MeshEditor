#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"

#include "Ray.h"

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
    void SetTargetDistance(float); // Start moving camera to provided distance.
    // Rotate camera around target by the given delta in spherical coordinates.
    // (azimuth, elevation) / (yaw, pitch) / (x, y) / (phi, theta)
    void Rotate(vec2 angles_delta);

    bool Tick(); // Move camera to target distance. Returns true if camera moved.
    void StopMoving(); // Stop target distance movement.

private:
    float TargetDistance;
    float TickSpeed{0.3};
    bool IsMoving{false};

    void SetDistance(float distance);
};
