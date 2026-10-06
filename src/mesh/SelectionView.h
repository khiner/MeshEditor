#pragma once

#include "gpu/MeshElementBlock.h"
#include "mesh/SelectionIndex.h"

#include <bit>
#include <optional>
#include <span>

// Borrowed canonical masks over one mesh domain's owned blocks, visited in ascending handle order.
struct SelectionView {
    std::span<const uint32_t> Bits; // The domain's canonical mask words.
    SelectionIndexView Tree;
    uint32_t Selected{};
    SelectionIndexMask Kind{SelectionIndexMask::Selected};

    uint32_t Count() const { return Selected; }
    bool Contains(uint32_t handle) const {
        return handle / 32u < Bits.size() && (Bits[handle / 32u] & (1u << (handle % 32u))) != 0u;
    }
    // Visits each block holding a selected element with the number of selected elements before it, and returns their total.
    uint32_t ForEachBlock(auto &&visit) const {
        if (!Selected) return 0u;
        uint32_t before = 0u;
        Tree.Visit(Kind, false, [&](uint32_t block) {
            visit(block, before);
            before += Kind == SelectionIndexMask::Hidden ? Tree.Leaves[block].Hidden : Tree.Leaves[block].Selected;
            return true;
        });
        return before;
    }
    void ForEach(auto &&visit) const {
        ForEachBlock([&](uint32_t block, uint32_t) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                for (auto bits = Bits[block * MeshElementBlockWords + w]; bits; bits &= bits - 1u)
                    visit(block * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits)));
        });
    }
    std::optional<uint32_t> Extreme(bool last) const {
        std::optional<uint32_t> result;
        Tree.Visit(Kind, last, [&](uint32_t block) {
            for (uint32_t i = 0u; i < MeshElementBlockWords; ++i) {
                const auto w = last ? MeshElementBlockWords - 1u - i : i;
                if (const auto bits = Bits[block * MeshElementBlockWords + w]) {
                    result = block * MeshElementBlockSize + w * 32u +
                        (last ? 31u - uint32_t(std::countl_zero(bits)) : uint32_t(std::countr_zero(bits)));
                    break;
                }
            }
            return false;
        });
        return result;
    }
    std::optional<uint32_t> First() const { return Extreme(false); }
    std::optional<uint32_t> Last() const { return Extreme(true); }
};

// The live edges of one mesh without an opposite halfedge.
// Only edge blocks whose aggregate carries the boundary flag are visited.
struct BoundaryEdgeView {
    SelectionIndexView Tree;
    std::span<const MeshElementBlock> Membership;
    std::span<const uint32_t> EdgeHalfedges, Opposites;
    uint32_t Owner{InvalidOffset};

    bool Contains(uint32_t edge) const {
        const auto block = edge / MeshElementBlockSize;
        if (block >= Membership.size() || Membership[block].Owner != Owner) return false;
        if (!(Membership[block].Live[(edge % MeshElementBlockSize) / 32u] & (1u << (edge % 32u)))) return false;
        const auto h = EdgeHalfedges[edge];
        return h != InvalidOffset && Opposites[h] == InvalidOffset;
    }
    void ForEach(auto &&visit) const {
        Tree.Visit(SelectionIndexMask::Boundary, false, [&](uint32_t block) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                for (auto bits = Membership[block].Live[w]; bits; bits &= bits - 1u) {
                    const auto edge = block * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
                    const auto h = EdgeHalfedges[edge];
                    if (h != InvalidOffset && Opposites[h] == InvalidOffset) visit(edge);
                }
            return true;
        });
    }
    uint32_t Count() const {
        uint32_t count = 0u;
        ForEach([&](uint32_t) { ++count; });
        return count;
    }
};
