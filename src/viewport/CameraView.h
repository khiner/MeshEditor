#pragma once

#include "CameraTypes.h"
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/ray.h"
#include "numeric/rect.h"

// Camera pose and lens, independent of navigation and animation.
struct CameraView {
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

    bool operator==(const CameraView &) const = default;
};
