#pragma once

#include "gpu/Types.h"

struct SolidLight {
    vec3 Direction DEFAULT();
    vec3 SpecularColor DEFAULT();
    vec3 DiffuseColor DEFAULT();
    float Wrap DEFAULT();
#ifndef __METAL_VERSION__
    bool operator==(const SolidLight &) const = default;
#endif
};
