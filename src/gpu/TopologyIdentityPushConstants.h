#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyJob.h"
#include "gpu/MeshTopologyPushConstants.h"
#include "gpu/SlotOffset.h"

struct TopologyIdentityPushConstants {
    MeshTopologyPushConstants Topology DEFAULT();
    MeshTopologyJob Job DEFAULT();
    GpuArray<SlotOffset, 2> Outputs DEFAULT(); // V, F
    GpuArray<ElementWork, 2> NewElements DEFAULT(); // Compact output ordinals
    GpuArray<ElementWork, 2> RetiredElements DEFAULT(); // Canonical source handles
    GpuArray<ElementWork, 2> ReplacedElements DEFAULT(); // Old corners and triangles
    SlotOffset Error DEFAULT();
    GpuArray<ElementHandleRange, 2> Inserted DEFAULT(); // The inserted handles of the new outputs, in compact order
};
static_assert(sizeof(TopologyIdentityPushConstants) == 1164);
