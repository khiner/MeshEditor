#pragma once
#include "gpu/Types.h"

// Uniform modes need no corner payload. Mixed meshes use canonical face
// sharpness and optional full-width sector-root handles at polygon corners.
enum class CornerClassMode : uint32_t {
    Mixed = 0u,
    UniformFace = 0xfffffffeu,
    UniformVertex = 0xffffffffu,
};
