#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// Edit Visibility over one record: change bits by handle for the source domain's newly hidden elements, then for the elements the hide reaches elsewhere.
// Domains are vertex, edge and face in that order, and masks are words by arena address.
// Each Changes run holds one word per 32 handles below the domain's bound.
struct EditVisibilityPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    GpuArray<DomainBlocks, 3> Blocks DEFAULT();
    GpuArray<uint32_t, 3> HiddenSlot DEFAULT(), SelectionSlot DEFAULT();
    GpuArray<uint32_t, 3> Changes DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run above offsets into
    uint32_t Source DEFAULT();
    uint32_t Unselected DEFAULT(); // Hides the unselected elements instead of the selected
};
