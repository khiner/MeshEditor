#pragma once

#include "numeric/VectorMath.h"

using numeric::vec3;

struct ray {
    vec3 o, d;
    vec3 operator()(float t) const { return o + d * t; }
};
