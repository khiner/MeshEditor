#pragma once

#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"

struct SilhouettePushConstants {
    VisibilityShadingPushConstants Visibility;
    // Opaque routes yield the pixels the visibility image assigns to another outlined object, which resolves coplanar ties and occlusion among outlines exactly.
    uint32_t YieldToOutlinedOwner DEFAULT();
};
static_assert(sizeof(SilhouettePushConstants) == sizeof(VisibilityShadingPushConstants) + 4, "SilhouettePushConstants size");
