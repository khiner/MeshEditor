#pragma once

#include "gpu/Element.h"
#include "gpu/Types.h"

struct EditSelectionSummary {
    vec3 PositionSum DEFAULT();
    Element Mode DEFAULT();
    uint32_t SelectedCount DEFAULT();
    uint32_t SelectedVertexCount DEFAULT();
    // Bit 0 means a selected sharp element exists. Bit 1 means a selected smooth element exists.
    uint32_t SharpnessFlags DEFAULT();
    uint32_t ActiveHandle DEFAULT(InvalidOffset);
};
static_assert(sizeof(EditSelectionSummary) == 32, "EditSelectionSummary size");
