#pragma once

#include "gpu/RenderTableRuns.h"

// Defines a producer-independent meshlet range with deterministic flattened WorkOffset ordering.
// Members names the leaf's first member handle, and the table runs resolve its record's handles.
struct MeshletWorkRange {
    uint32_t Instance DEFAULT();
    uint32_t Members DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t WorkOffset DEFAULT();
    RenderTableRuns Tables DEFAULT();
};
