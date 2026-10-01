#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

struct CornerClassificationPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    ElementWork Vertices DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t CornerBlocksSlot DEFAULT(), CornerOwner DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT(), FaceSharpnessSlot DEFAULT();
    ElementAttributeRef CornerSectors DEFAULT();
    ElementWork DirtyBlocks DEFAULT(), NeededBlocks DEFAULT();
    SlotOffset State DEFAULT(); // Incoming count, then face/edge flags
    uint32_t StatusOffset DEFAULT(); // One classification-presence word per dirty block, from State
};
static_assert(sizeof(CornerClassificationPushConstants) == 152);
