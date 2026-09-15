#pragma once

#include "gpu/Types.h"

struct MeshletCullPushConstants {
    uint32_t InstanceCount DEFAULT();
    uint32_t WorkBlockCount DEFAULT();
    uint32_t WorkRangeSlot DEFAULT();
    uint32_t WorkBlockSlot DEFAULT();
    // Traversal node arena, ping-pong frontiers, per-level sizes, and per-block prefixes.
    uint32_t LodNodeSlot DEFAULT();
    uint32_t LodFrontierSlot DEFAULT();
    uint32_t LodFrontierAltSlot DEFAULT();
    uint32_t LodFrontierStateSlot DEFAULT();
    uint32_t LodFrontierBlockStateSlot DEFAULT();
    uint32_t LodExpandArgsSlot DEFAULT();
    // Selects the input frontier buffer and state.
    uint32_t LodFrontierIndex DEFAULT();
    // Marks the seed level, which derives its frontier from instance IDs.
    uint32_t LodSeedLevel DEFAULT();
    // Marks the final level, which emits meshlet ranges.
    uint32_t LodFinalLevel DEFAULT();
    uint32_t WorkStateSlot DEFAULT();
    uint32_t WorkDispatchArgsSlot DEFAULT();
    uint32_t BlockStateSlot DEFAULT();
    uint32_t ClassificationSlot DEFAULT();
    uint32_t VisibleSlot DEFAULT();
    uint32_t InstanceMapSlot DEFAULT();
    uint32_t InstanceSlot DEFAULT();
    uint32_t PrimitiveSlot DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t ClusterGroupSlot DEFAULT(InvalidSlot);
    uint32_t BoundsSlot DEFAULT();
    uint32_t ModelSlot DEFAULT();
    uint32_t PosedMeshletBoundsSlot DEFAULT();
    uint32_t RouteStateSlot DEFAULT();
    uint32_t DispatchArgsSlot DEFAULT();
    uint32_t DispatchChunkCount DEFAULT();
    uint32_t DispatchChunkSize DEFAULT();
    uint32_t RouteMode DEFAULT();
    uint32_t RequiredInstanceFlags DEFAULT();
    uint32_t RouteMask DEFAULT(0x1ffu);
    uint32_t PyramidSamplerSlot DEFAULT(InvalidSlot);
    // Counts coarse clusters selected across classification blocks.
    uint32_t CoarseCountSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshletCullPushConstants) == 140, "MeshletCullPushConstants size");
