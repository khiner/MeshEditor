#pragma once

#include "vec2.h"

struct rect {
    vec2 pos, size;
    vec2 max() const { return pos + size; }
};
