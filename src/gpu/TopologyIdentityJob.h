#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"

// One edit's job in a pass of its output identities.
// Scratch is the batch's scratch, whose storage slot holds the edit's MeshTopologyJob record at word Job.
// ErrorSlot holds the error word at its start.
struct TopologyIdentityJob {
    uint32_t Job DEFAULT();
    SlotOffset Scratch DEFAULT();
    uint32_t FaceTriangleStartSlot DEFAULT(InvalidSlot); // The source's
    uint32_t ErrorSlot DEFAULT(InvalidSlot);
    GpuArray<SlotOffset, 2> Outputs DEFAULT(); // V, F
    GpuArray<ElementWork, 2> NewElements DEFAULT(); // Compact output ordinals
    GpuArray<ElementWork, 2> RetiredElements DEFAULT(); // Source handles
    GpuArray<ElementWork, 2> ReplacedElements DEFAULT(); // Old corners and triangles
    GpuArray<ElementHandleRange, 2> Inserted DEFAULT(); // The inserted handles of the new outputs, in compact order
};
