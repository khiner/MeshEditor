#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// Marks, per affected vertex or edge of a topology edit, whether it now draws as a point or line without an owner cluster, or no longer draws while it has one.
// Domains are vertices then edges, and each bit run holds one word per 32 handles below the larger of the domain's two bounds.
struct ElementRepairPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    GpuArray<uint32_t, 2> MembershipSlot DEFAULT();
    GpuArray<uint32_t, 2> Bound DEFAULT(); // Handles the record's tables cover, below which a handle can be live
    GpuArray<SlotOffset, 2> Owners DEFAULT(); // The record's point and line owner layers by handle, or no slot
    GpuArray<uint32_t, 2> OwnerBound DEFAULT(); // Handles the owner layers cover, which retired handles beyond the tables may still have
    GpuArray<uint32_t, 2> Added DEFAULT(), Removed DEFAULT();
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer the bit runs offset into
    ElementWork Work DEFAULT();
    uint32_t WorkCount DEFAULT(); // One lane per slot of the work's occupied blocks
    uint32_t WorkDomain DEFAULT(); // 0 vertices, 1 edges, 2 halfedges, which mark their edges
};
