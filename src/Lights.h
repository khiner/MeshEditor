#pragma once

#include "numeric/vec3.h"

struct Lights {
    vec3 ViewColor; // Light emitting from the view position
    float AmbientIntensity;
    vec3 DirectionalColor;
    float DirectionalIntensity;
    vec3 Direction;
};
