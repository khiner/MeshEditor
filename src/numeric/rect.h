#pragma once

#include "numeric/VectorMath.h"

using numeric::vec2;

struct rect {
    vec2 pos, size;
    vec2 max() const { return pos + size; }
};
