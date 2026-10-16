#pragma once

#include "gpu/Types.h"

// Stores positions and point-domain colors. Triangle tangent, color, and UV attributes use per-corner arenas.
struct Vertex {
    vec3 Position DEFAULT();
};
