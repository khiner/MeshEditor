#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementLayers.h"
#include "gpu/MeshDisplay.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Stores one mesh's display state and arena locations once, shared by every primitive and instance that draws it.
// The connectivity's tables resolve the mesh's record-relative handles to arena addresses.
struct MeshRecord {
    MeshDisplay Display DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT(); // Corner vertex handles, or an extras record's bone and joint index run
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassMode DEFAULT(InvalidOffset);
    ConnectivityRef Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    uint32_t MorphShadingAuthored DEFAULT();
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    ElementLayers Layers DEFAULT();
    // Rounds the record to whole cache lines, so each record's display fields start one.
    GpuArray<uint32_t, 6> Padding DEFAULT();
};
