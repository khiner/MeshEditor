#pragma once

#include "gpu/Transform.h"

// Distinct component type so an entity can hold both local and world transforms.
struct WorldTransform : Transform {
    using Transform::Transform;
    WorldTransform(const Transform &t) : Transform{t} {}
};
// World-space transform, composed from the local Transform and parent chain.

// World-space forward direction for a camera entity (matches glTF: cameras look down -Z).
inline vec3 CameraForward(const WorldTransform &wt) {
    return -glm::normalize(glm::rotate(wt.R, vec3{0.f, 0.f, 1.f}));
}
