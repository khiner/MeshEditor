#pragma once

#include "viewport/CameraView.h"

// Uses the viewport aspect ratio rather than the source camera's aspect ratio.
struct ViewCamera : CameraView {
    ViewCamera(vec3 position, vec3 target, Camera data)
        : CameraView{data, target, numeric::Length(position - target), OrientationFromAway(position - target)} {}

    ViewCamera(vec3 position, quat orientation, Camera data)
        : CameraView{data, {}, 1.f, numeric::Normalize(orientation)} {
        Target = position - Orientation * vec3{0, 0, 1};
    }

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
