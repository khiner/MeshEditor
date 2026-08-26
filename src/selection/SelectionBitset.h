#pragma once

#include "gpu/Element.h"

#include <span>
#include <type_traits>
#include <vector>

struct Mesh;

namespace selection {

// Every function here takes one mesh's own selection-bit words, where element `handle` is bit
// `handle % 32` of word `handle / 32`, and `count`, the element count of the current edit mode.
// The store sizes a mesh's bits to its largest element domain, so `count` never exceeds them.

// Set the first `count` bits, clearing any gap bits in the last word.
void SelectAll(std::span<uint32_t> bits, uint32_t count);
uint32_t CountSelected(std::span<const uint32_t> bits, uint32_t count);
inline bool IsSelected(std::span<const uint32_t> bits, uint32_t handle) { return (bits[handle >> 5] >> (handle & 31u)) & 1u; }
inline void Select(std::span<uint32_t> bits, uint32_t handle) { bits[handle >> 5] |= 1u << (handle & 31u); }
inline void Deselect(std::span<uint32_t> bits, uint32_t handle) { bits[handle >> 5] &= ~(1u << (handle & 31u)); }
void ForEachSelected(std::span<const uint32_t> bits, uint32_t count, auto &&fn) {
    const uint32_t last_word = (count + 31) / 32;
    for (uint32_t w = 0; w < last_word; ++w) {
        uint32_t word = bits[w];
        while (word) {
            const uint32_t handle = w * 32 + __builtin_ctz(word);
            if (handle < count) fn(handle);
            word &= word - 1;
        }
    }
}
// Visit every edge with an endpoint among the selected vertices.
// An edge with both endpoints selected fires once, from its lower-indexed endpoint.
void ForEachVertexTouchedEdge(std::span<const uint32_t> bits, uint32_t count, const auto &mesh, auto &&fn) {
    const auto adjacency = mesh.GetVertexEdgeAdjacency();
    if (adjacency.Offsets.empty()) return;
    using MeshT = std::remove_cvref_t<decltype(mesh)>;
    ForEachSelected(bits, count, [&](uint32_t v) {
        for (const auto e : adjacency.Incident(v)) {
            const auto hh = mesh.GetHalfedge(typename MeshT::EH{e}, 0);
            const auto from = mesh.GetFromVertex(hh);
            const auto other = from && *from == v ? mesh.GetToVertex(hh) : from;
            if (other && *other < v && *other < count && IsSelected(bits, *other)) continue;
            fn(e);
        }
    });
}
// Convert the selected `from_element` handles to `to_element` handles.
// Matching elements return the selected handles as they are.
std::vector<uint32_t> ConvertSelectionElement(std::span<const uint32_t> bits, uint32_t count, const Mesh &, Element from_element, Element to_element);
uint32_t GetElementCount(const Mesh &, Element);

} // namespace selection
