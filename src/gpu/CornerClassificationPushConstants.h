#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

struct CornerClassificationPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    ElementWork Vertices DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT(), FaceSharpnessSlot DEFAULT();
    SlotOffset CornerSectors DEFAULT(); // The record's sector roots, without a slot while it holds none
    ElementWork DirtyBlocks DEFAULT(), NeededBlocks DEFAULT();
    SlotOffset State DEFAULT(); // Incoming count, then face/edge flags
};
static_assert(sizeof(CornerClassificationPushConstants) == 152);
