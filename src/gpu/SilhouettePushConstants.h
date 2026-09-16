#pragma once

#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"

struct SilhouettePushConstants {
    VisibilityShadingPushConstants Visibility;
    // Opaque routes outline only the pixels the visibility image assigns to the same object, which keeps coplanar ties out.
    uint32_t RequireOwner DEFAULT();
};
static_assert(sizeof(SilhouettePushConstants) == sizeof(VisibilityShadingPushConstants) + 4, "SilhouettePushConstants size");
