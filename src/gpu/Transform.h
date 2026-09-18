#pragma once

#include "gpu/Types.h"
#ifndef __METAL_VERSION__
#include "FieldLimits.h"
#endif

struct Transform {
    vec3 P DEFAULT();
    quat R DEFAULT(1, 0, 0, 0);
    vec3 S DEFAULT(1, 1, 1);
#ifndef __METAL_VERSION__
    bool operator==(const Transform &) const = default;
#endif
};
static_assert(sizeof(Transform) == 40, "Transform size");
static_assert(alignof(Transform) == 4 && __builtin_offsetof(Transform, P) == 0 && __builtin_offsetof(Transform, R) == 12 && __builtin_offsetof(Transform, S) == 28);
#ifndef __METAL_VERSION__
template<> struct FieldLimits<&Transform::S> : Within<0.01f, 10.f> {};
#endif
