#pragma once

#include "mat4.h"
#include "vec3.h"

struct Ray {
    const vec3 Origin, Direction;

    vec3 operator()(float t) const { return Origin + Direction * t; }

    Ray WorldToLocal(const mat4 &model) const;
};
