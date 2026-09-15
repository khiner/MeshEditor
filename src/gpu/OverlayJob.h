#pragma once

#include "gpu/Types.h"
#include "gpu/OverlayJobKind.h"
#include "gpu/ExtrasLineKind.h"

// Defines one persistent single-threadgroup procedural-line chunk with common visibility and indirect submission.
struct OverlayJob {
    OverlayJobKind Kind DEFAULT();
    uint32_t InstanceIndex DEFAULT();
    uint32_t FirstElement DEFAULT();
    uint32_t ElementCount DEFAULT();
    uint32_t SourceOffset DEFAULT();
    uint32_t IndexOffset DEFAULT();
    ExtrasLineKind ExtrasKind DEFAULT();
    vec3 LocalOffset DEFAULT();
    vec4 Params DEFAULT();
};
static_assert(sizeof(OverlayJob) == 56, "OverlayJob size");
