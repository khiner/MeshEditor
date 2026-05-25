#pragma once

#include "gpu/Element.h"

#include <cstdint>
#include <span>
#include <vector>

struct Mesh;

// Per-mesh offset/count into SelectionBitsetBuffer for the current edit element type.
// Assigned on Edit mode entry; updated on element type switch and mesh topology change.
struct MeshSelectionBitsetRange {
    uint32_t Offset; // Start bit index in SelectionBitsetBuffer
    uint32_t Count; // Element count for current edit mode
};

namespace selection {

// Set all bits in [offset, offset+count), clearing any gap bits in the last word.
void SelectAll(uint32_t *bits, uint32_t offset, uint32_t count);
// Count selected bits in [offset, offset+count).
uint32_t CountSelected(const uint32_t *bits, uint32_t offset, uint32_t count);
// Return local (0-based) handles of all set bits in [offset, offset+count).
std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count);
std::vector<uint32_t> ConvertSelectionElement(std::span<const uint32_t> handles, const Mesh &, Element from_element, Element to_element);
uint32_t GetElementCount(const Mesh &, Element);

} // namespace selection
