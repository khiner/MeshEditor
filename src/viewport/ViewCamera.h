#pragma once

#include "CameraTypes.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/rect.h"

// The viewport navigation camera (orbit/zoom around a target point).
// Orientation is a single world-space rotation (camera-local -> world: +X right, +Y up, +Z away from the view direction).
// Note: Aspect ratio of the provided data is ignored, as the ViewCamera follows the viewport aspect ratio.
struct ViewCamera {
    ViewCamera(vec3 position, vec3 target, Camera data)
        : Data{data}, Target{target}, Distance{glm::length(position - target)}, Orientation{OrientationFromAway(position - target)} {}

    ViewCamera(vec3 position, quat orientation, Camera data)
        : Data{data}, Distance{1.f}, Orientation{glm::normalize(orientation)} {
        Target = position - Orientation * vec3{0, 0, 1}; // Distance 1, so Position() == position.
    }

    Camera Data;
    vec3 Target;
    float Distance;
    quat Orientation; // World rotation; columns are Right, Up, Away (Away == backward of the view direction).

    float NearClip() const;
    float FarClip() const; // Always finite (fallback when perspective far is infinite).

    vec3 Forward() const { return Orientation * vec3{0, 0, 1}; } // "Away": points from Target toward the camera.
    vec3 Up() const { return Orientation * vec3{0, 1, 0}; }
    mat3 Basis() const; // Right, Up, -Forward
    ray Ray() const { return {Position(), Forward()}; }
    mat4 View() const;
    mat4 Projection(float aspect_ratio) const;
    vec3 Position() const { return Target + Distance * Forward(); }
    ray PixelToWorldRay(vec2 mouse_px, rect viewport) const;

    bool IsAligned(vec3 direction) const;
    bool IsInFront(vec3) const;

    // Direct (non-animated) mutators for interactive input. Cancel any in-flight transition.
    void RotateBy(vec2 yaw_pitch_delta);
    void ZoomBy(float factor); // Multiplies Distance.

    // Smoothstep ease-in/out over a fixed tick count.
    void AnimateTo(vec3 target, quat orientation, float distance);
    void SetTargetDirection(vec3 away); // Animate to face along `away` with a level horizon.
    // Animate (slerp) into looking through a scene camera at `camera_position` with world rotation `orientation`.
    void AnimateToLookThrough(vec3 camera_position, quat orientation, float distance);

    bool IsAnimating() const { return Anim.has_value(); }
    void StopMoving() { Anim.reset(); }

    bool Tick();

    // World rotation whose +Z (Away) points along `away`, with no roll (level horizon).
    static quat OrientationFromAway(vec3 away);

private:
    struct Animation {
        vec3 SrcTarget, DstTarget;
        float SrcDistance, DstDistance;
        quat SrcOrientation, DstOrientation; // DstOrientation hemisphere-aligned to Src for a shortest-path slerp.
        uint32_t Frame; // Incremented each Tick; animation completes at DurationFrames.
    };
    std::optional<Animation> Anim{};

    void ApplyDistance(float new_distance); // Updates Distance and scales orthographic Mag in lockstep.
};

// At most one camera carries this component at a time.
struct LookingThrough {
    ViewCamera SavedViewCamera; // The pre-look-through ViewCamera, restored on exit.
};
