#pragma once

#include "gpu/Types.h"

struct SolidLight {
    vec3 Direction DEFAULT();
    vec3 SpecularColor DEFAULT();
    vec3 DiffuseColor DEFAULT();
    float Wrap DEFAULT();
};
static_assert(sizeof(SolidLight) == 40, "SolidLight size");
