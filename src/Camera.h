#pragma once

#include "Ray.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

static const glm::mat4 I{1};
static const glm::vec3 Origin{0, 0, 0};
static const glm::vec3 Up{0, 1, 0};

struct Camera {
    Camera(const glm::vec3 &position, const glm::vec3 &target, float field_of_view, float near_clip, float far_clip)
        : Position(position), Target(target), FieldOfView(field_of_view), NearClip(near_clip), FarClip(far_clip) {}
    ~Camera() = default;

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspect_ratio) const;
    glm::mat4 GetInvViewProjectionMatrix(float aspect_ratio) const;
    float GetDistance() const;

    // Converts a position in clip space to a ray in world space.
    Ray ClipPosToWorldRay(const glm::vec2 &pos_clip, float aspect_ratio) const;

    void SetPositionFromView(const glm::mat4 &view);
    void SetTargetDistance(float distance); // Start moving camera to provided distance.

    bool Tick(); // Move camera to target distance. Returns true if camera moved.
    void StopMoving(); // Stop target distance movement.

    glm::vec3 Position;
    glm::vec3 Target;
    float FieldOfView;
    float NearClip, FarClip;

private:
    void SetDistance(float distance);

    float TargetDistance;
    float TickSpeed{0.3};
    bool IsMoving{false};
};
