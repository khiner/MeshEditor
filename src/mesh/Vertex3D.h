#pragma once

#include "numeric/vec3.h"

// Shared CPU/GPU vertex layout.
struct Vertex3D {
    vec3 Position;
    vec3 Normal;
};
