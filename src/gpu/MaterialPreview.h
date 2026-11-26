#pragma once

#include "gpu/Types.h"

// A corner of the preview sphere's triangles, three per triangle, in world space, which is its object's space.
// The tangent's sign orients the bitangent, and the texture coordinates have V down.
struct PreviewCorner {
    vec3 Position, Normal, Tangent;
    float TangentSign;
    vec2 Uv;
};

// The argument table index of the preview sphere's corners.
GPU_CONSTANT uint32_t BufferIndex_PreviewCorners = 6u;

// How a preview's display shows its image: a lit material's radiance, a value's color, or a vector's components mapped from [-1, 1].
enum class PreviewDisplay : uint32_t {
    Lit,
    Color,
    Vector,
};
struct MaterialPreviewPushConstants {
    uint32_t Material DEFAULT(); // The material record the preview shades
    PreviewDisplay Display DEFAULT(PreviewDisplay::Lit);
};
