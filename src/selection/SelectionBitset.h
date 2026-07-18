#pragma once

#include "gpu/Element.h"

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
// A mesh's element handles map to consecutive global bits starting at its range `offset`,
// packed 32 per word: element `local`'s bit is bit (offset + local) % 32 of word (offset + local) / 32.
inline bool IsSelected(const uint32_t *bits, uint32_t offset, uint32_t local) {
    const uint32_t i = offset + local;
    return (bits[i >> 5] >> (i & 31u)) & 1u;
}
inline void Select(uint32_t *bits, uint32_t offset, uint32_t local) {
    const uint32_t i = offset + local;
    bits[i >> 5] |= 1u << (i & 31u);
}
inline void Deselect(uint32_t *bits, uint32_t offset, uint32_t local) {
    const uint32_t i = offset + local;
    bits[i >> 5] &= ~(1u << (i & 31u));
}
// Visit the local (0-based) handle of every set bit in [offset, offset+count).
void ForEachSelected(const uint32_t *bits, uint32_t offset, uint32_t count, auto &&fn) {
    const uint32_t first_word = offset / 32, last_word = (offset + count + 31) / 32;
    for (uint32_t w = first_word; w < last_word; ++w) {
        uint32_t word = bits[w];
        while (word) {
            const uint32_t global_idx = w * 32 + __builtin_ctz(word);
            if (global_idx >= offset && global_idx < offset + count) fn(global_idx - offset);
            word &= word - 1;
        }
    }
}
// Visit every edge with an endpoint among the selected vertices in [offset, offset+count).
void ForEachVertexTouchedEdge(const uint32_t *bits, uint32_t offset, uint32_t count, const auto &mesh, auto &&fn) {
    for (const auto eh : mesh.edges()) {
        const auto heh = mesh.GetHalfedge(eh, 0);
        const auto from = mesh.GetFromVertex(heh), to = mesh.GetToVertex(heh);
        if ((from && *from < count && IsSelected(bits, offset, *from)) || (to && *to < count && IsSelected(bits, offset, *to))) fn(*eh);
    }
}
// Return local (0-based) handles of all set bits in [offset, offset+count).
std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count);
// Convert the selected `from_element` handles in [offset, offset+count) to `to_element` handles.
std::vector<uint32_t> ConvertSelectionElement(const uint32_t *bits, uint32_t offset, uint32_t count, const Mesh &, Element from_element, Element to_element);
uint32_t GetElementCount(const Mesh &, Element);

} // namespace selection
