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
inline bool Empty(const DomainSelection &selection) { return selection.Handles.empty() && selection.Blocks.empty(); }

// The selection of the ascending local blocks with their nonempty masks.
inline DomainSelection CompactSelection(std::vector<SelectedBlock> blocks) {
    uint64_t count = 0u;
    for (const auto &[local, words] : blocks) {
        for (const auto word : words) count += uint64_t(std::popcount(word));
    }
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
    for (const auto handle : handles) {
        const auto local = handle / MeshElementBlockSize, slot = handle % MeshElementBlockSize;
        if (blocks.empty() || blocks.back().Local != local) blocks.push_back({local, SelectionBlock{}});
        blocks.back().Mask[slot / 32u] |= 1u << (slot % 32u);
    }
    return CompactSelection(std::move(blocks));
}
// Calls visit(local block, mask) for each local block holding a selected handle, in ascending order.
void ForEachSelectedBlock(const DomainSelection &selection, auto &&visit) {
    for (const auto &[local, words] : selection.Blocks) visit(local, words);
    for (size_t i = 0u; i < selection.Handles.size();) {
        const auto local = selection.Handles[i] / MeshElementBlockSize;
        SelectionBlock words{};
        for (; i < selection.Handles.size() && selection.Handles[i] / MeshElementBlockSize == local; ++i) {
            const auto slot = selection.Handles[i] % MeshElementBlockSize;
            words[slot / 32u] |= 1u << (slot % 32u);
        }
        visit(local, words);
    }
}
inline uint32_t SelectedCount(const DomainSelection &selection) {
    uint32_t count = uint32_t(selection.Handles.size());
    for (const auto &[local, words] : selection.Blocks) {
        for (const auto word : words) count += uint32_t(std::popcount(word));
    }
    return count;
}

// The selected handles of a node's input record, the edit domain they were captured in, and the active element's handle.
// An empty selection selects everything.
struct InlineSelection {
    DomainSelection Vertices, Edges, Faces;
    Element Domain{Element::Vertex};
    uint32_t Active{InvalidOffset};
};
inline bool Empty(const InlineSelection &selection) { return Empty(selection.Vertices) && Empty(selection.Edges) && Empty(selection.Faces); }
// The inline selection of the handles, captured in the edit domain.
inline InlineSelection SelectionOf(const GeometrySelection &handles, Element domain) {
    return {.Vertices = SelectionOf(handles.Vertices), .Edges = SelectionOf(handles.Edges), .Faces = SelectionOf(handles.Faces), .Domain = domain};
}
} // namespace procedure
