#pragma once

#include "numeric/vec3.h"
#include "numeric/vec4.h"

// These are submitted to shaders.
struct Vertex3D {
    vec3 Position;
    vec3 Normal;
    vec4 Color;
};
