#pragma once

#include "gpu/Types.h"

// Configures deterministic compaction of enabled persistent procedural-line jobs.
struct OverlayJobCullPushConstants {
    uint32_t JobsSlot DEFAULT();
    uint32_t JobCount DEFAULT();
    uint32_t InstanceStateSlot DEFAULT();
    uint32_t BlockStateSlot DEFAULT();
    uint32_t VisibleSlot DEFAULT();
    uint32_t DispatchArgsSlot DEFAULT();
    uint32_t ExtrasOnly DEFAULT();
};
static_assert(sizeof(OverlayJobCullPushConstants) == 28, "OverlayJobCullPushConstants size");
