#pragma once

#include "Camera.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/ray.h"
#include "numeric/rect.h"
#include "numeric/vec3.h"

#include <optional>

// The viewport navigation camera (orbit/zoom around a target point).
// Note: Aspect ratio of the provided data is ignored, as the ViewCamera follows the viewport aspect ratio.
struct ViewCamera {
    ViewCamera(vec3 position, vec3 target, Camera data)
        : Data{data}, Target{target}, Distance{glm::length(position - target)} {
        const auto direction = glm::normalize(position - target);
        YawPitch = {atan2(direction.z, direction.x), asin(direction.y)};
    }

    Camera Data;
    vec3 Target;
    float Distance;
    vec2 YawPitch; // Ranges ([0, 2π], [-π, π]) If pitch is in (wrapped) range [π/2, 3π/2], camera is flipped

    float NearClip() const;
    float FarClip() const; // Always finite (fallback when perspective far is infinite).

    vec3 Forward() const;
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
    void SetTargetDirection(vec3);
    void AnimateTo(vec3 target, vec2 yaw_pitch, float distance);

    bool IsAnimating() const { return Anim.has_value(); }
    void StopMoving() { Anim.reset(); }

    bool Tick();

private:
    struct Animation {
        vec3 SrcTarget, DstTarget;
        float SrcDistance, DstDistance;
        vec2 SrcYawPitch, DstYawPitch; // DstYawPitch unwrapped relative to Src for shortest-path lerp.
        uint32_t Frame; // Incremented each Tick; animation completes at DurationFrames.
    };
    std::optional<Animation> Anim{};

    void ApplyDistance(float new_distance); // Updates Distance and scales orthographic Mag in lockstep.
    vec3 YAxis() const; // May be flipped depending on pitch.
};
