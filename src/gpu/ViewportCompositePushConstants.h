#pragma once

#include "gpu/Types.h"

struct ViewportCompositePushConstants {
    uint32_t SceneColorSamplerSlot DEFAULT();
    uint32_t OverlayColorSamplerSlot DEFAULT();
    // View transform 0 encodes, 1 tone maps and encodes, and 2 preserves debug values.
    uint32_t ViewTransform DEFAULT();
    uint32_t HasOverlay DEFAULT();
    vec4 Backdrop DEFAULT();
};
static_assert(sizeof(ViewportCompositePushConstants) == 32, "ViewportCompositePushConstants size");
