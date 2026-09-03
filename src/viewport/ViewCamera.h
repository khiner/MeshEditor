#pragma once

#include "CameraTypes.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/rect.h"

// Uses the viewport aspect ratio rather than the source camera's aspect ratio.
struct ViewCamera {
    ViewCamera(vec3 position, vec3 target, Camera data)
        : Data{data}, Target{target}, Distance{numeric::Length(position - target)}, Orientation{OrientationFromAway(position - target)} {}

    ViewCamera(vec3 position, quat orientation, Camera data)
        : Data{data}, Distance{1.f}, Orientation{numeric::Normalize(orientation)} {
        Target = position - Orientation * vec3{0, 0, 1};
    }

    Camera Data;
    vec3 Target;
    float Distance;
    quat Orientation;

    float NearClip() const;
    // Returns a finite fallback for an infinite perspective far plane.
    float FarClip() const;

    vec3 Forward() const { return Orientation * vec3{0, 0, 1}; }
    vec3 Up() const { return Orientation * vec3{0, 1, 0}; }
    mat3 Basis() const;
    ray Ray() const { return {Position(), Forward()}; }
    mat4 View() const;
    mat4 Projection(float aspect_ratio) const;
    vec3 Position() const { return Target + Distance * Forward(); }
    ray PixelToWorldRay(vec2 mouse_px, rect viewport) const;

    bool IsAligned(vec3 direction) const;
    bool IsInFront(vec3) const;

    // Interactive changes cancel an active transition.
    void RotateBy(vec2 yaw_pitch_delta);
    void ZoomBy(float factor);

    void AnimateTo(vec3 target, quat orientation, float distance);
    void SetTargetDirection(vec3 away);
    void AnimateToLookThrough(vec3 camera_position, quat orientation, float distance);

    bool IsAnimating() const { return Anim.has_value(); }
    void StopMoving() { Anim.reset(); }

    bool Tick();

    // Returns a level world rotation whose positive Z axis follows `away`.
    static quat OrientationFromAway(vec3 away);

private:
    struct Animation {
        vec3 SrcTarget, DstTarget;
        float SrcDistance, DstDistance;
        quat SrcOrientation, DstOrientation;
        uint32_t Frame;
    };
    std::optional<Animation> Anim{};

    void ApplyDistance(float new_distance);
};

struct LookingThrough {
    ViewCamera SavedViewCamera;
};
