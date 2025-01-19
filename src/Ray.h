#pragma once

#include "numeric/mat4.h"
#include "numeric/vec3.h"

struct Ray {
    const vec3 Origin, Direction;

    vec3 operator()(float t) const { return Origin + Direction * t; }

    // We already cash the transpose of the inverse transform for all models,
    // so save some compute by using that.
    Ray WorldToLocal(const mat4 &transp_inv_transform) const;
};
