#pragma once

#include "mesh/GeometrySelection.h"

#include <bit>

namespace procedure {
// A local block of a record's handles in one domain and the mask of its selected slots.
struct SelectedBlock {
    uint32_t Local;
    SelectionBlock Mask;
};
// One element domain's selected handles of a node's input record, as ascending handles or as the ascending local blocks holding a selected handle with their masks, whichever is smaller.
struct DomainSelection {
    std::vector<uint32_t> Handles;
    std::vector<SelectedBlock> Blocks;
};

// The number of selected slots in the blocks' masks.
inline uint32_t SelectedCount(std::span<const SelectedBlock> blocks) {
    uint32_t count = 0u;
    for (const auto &[local, words] : blocks) {
        for (const auto word : words) count += uint32_t(std::popcount(word));
    }
    return count;
}
inline uint32_t SelectedCount(const DomainSelection &selection) { return uint32_t(selection.Handles.size()) + SelectedCount(selection.Blocks); }
// Calls visit(local block, mask) for each local block holding one of the ascending handles, in ascending order.
void ForEachHandleBlock(std::span<const uint32_t> handles, auto &&visit) {
    for (size_t i = 0u; i < handles.size();) {
        const auto local = handles[i] / MeshElementBlockSize;
        SelectionBlock words{};
        for (; i < handles.size() && handles[i] / MeshElementBlockSize == local; ++i) {
            const auto slot = handles[i] % MeshElementBlockSize;
            words[slot / 32u] |= 1u << (slot % 32u);
        }
        visit(local, words);
    }
}

// The selection of the ascending local blocks with their nonempty masks.
inline DomainSelection CompactSelection(std::vector<SelectedBlock> blocks) {
    const auto count = SelectedCount(blocks);
    if (count * sizeof(uint32_t) > blocks.size() * sizeof(blocks.front())) return {.Blocks = std::move(blocks)};
    DomainSelection selection;
    selection.Handles.reserve(count);
    for (const auto &[local, words] : blocks) {
        for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
            for (auto bits = words[w]; bits; bits &= bits - 1u) selection.Handles.push_back(local * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits)));
        }
    }
    return selection;
}
// The selection of ascending handles.
inline DomainSelection SelectionOf(std::span<const uint32_t> handles) {
    std::vector<SelectedBlock> blocks;
    ForEachHandleBlock(handles, [&](uint32_t local, const SelectionBlock &mask) { blocks.push_back({local, mask}); });
    return CompactSelection(std::move(blocks));
}
// Calls visit(local block, mask) for each local block holding a selected handle, in ascending order.
void ForEachSelectedBlock(const DomainSelection &selection, auto &&visit) {
    for (const auto &[local, words] : selection.Blocks) visit(local, words);
    ForEachHandleBlock(selection.Handles, visit);
}

// The selected handles of a node's input record, the edit domain they were captured in, and the active element's handle.
// All selects every element of the input record, as does a node without an inline selection.
struct InlineSelection {
    DomainSelection Vertices, Edges, Faces;
    Element Domain{Element::Vertex};
    uint32_t Active{InvalidOffset};
    bool All{};
};
// The inline selection of the handles, captured in the edit domain.
inline InlineSelection SelectionOf(const GeometrySelection &handles, Element domain) {
    return {.Vertices = SelectionOf(handles.Vertices), .Edges = SelectionOf(handles.Edges), .Faces = SelectionOf(handles.Faces), .Domain = domain};
}
} // namespace procedure
