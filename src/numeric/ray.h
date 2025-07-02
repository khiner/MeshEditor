#pragma once

#include "numeric/vec3.h"

struct ray {
    vec3 o, d;
    vec3 operator()(float t) const { return o + d * t; }
};
