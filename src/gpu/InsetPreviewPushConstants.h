#pragma once

#include "gpu/SlotOffset.h"

// Each basis names an output vertex handle, which VertexTable resolves.
struct InsetPreviewPushConstants {
    SlotOffset Basis DEFAULT();
    SlotOffset VertexTable DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t Count DEFAULT();
    float Thickness DEFAULT(), Depth DEFAULT();
};
