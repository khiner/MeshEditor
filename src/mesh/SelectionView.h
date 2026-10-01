#pragma once

#include "gpu/MeshElementBlock.h"
#include "gpu/SelectionAggregate.h"

#include <bit>
#include <optional>
#include <span>

// Borrowed canonical masks over one mesh domain's owned blocks, visited in ascending handle order.
struct SelectionView {
    std::span<const uint32_t> Bits; // The domain's canonical mask words.
    std::span<const uint32_t> Blocks; // The mesh's owned blocks in ascending order.
    std::span<const SelectionAggregate> Leaves; // The domain's block aggregates.
    uint32_t Selected{};

    uint32_t Count() const { return Selected; }
    bool Contains(uint32_t handle) const {
        return handle / 32u < Bits.size() && (Bits[handle / 32u] & (1u << (handle % 32u))) != 0u;
    }
    // Visits each block holding a selected element with the number of selected elements before it, and returns their total.
    uint32_t ForEachBlock(auto &&visit) const {
        if (!Selected) return 0u;
        uint32_t before = 0u;
        for (const auto block : Blocks) {
            const auto count = Leaves[block].Selected;
            if (!count) continue;
            visit(block, before);
            before += count;
            if (before == Selected) break;
        }
        return before;
    }
    void ForEach(auto &&visit) const {
        ForEachBlock([&](uint32_t block, uint32_t) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                for (auto bits = Bits[block * MeshElementBlockWords + w]; bits; bits &= bits - 1u)
                    visit(block * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits)));
        });
    }
    std::optional<uint32_t> First() const {
        if (!Selected) return {};
        for (const auto block : Blocks) {
            if (!Leaves[block].Selected) continue;
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                if (const auto bits = Bits[block * MeshElementBlockWords + w]) return block * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
        }
        return {};
    }
    std::optional<uint32_t> Last() const {
        if (!Selected) return {};
        for (auto block = Blocks.rbegin(); block != Blocks.rend(); ++block) {
            if (!Leaves[*block].Selected) continue;
            for (uint32_t w = MeshElementBlockWords; w--;)
                if (const auto bits = Bits[*block * MeshElementBlockWords + w]) return *block * MeshElementBlockSize + w * 32u + 31u - uint32_t(std::countl_zero(bits));
        }
        return {};
    }
};

// The live edges of one mesh without an opposite halfedge.
// Only edge blocks whose aggregate carries the boundary flag are visited.
struct BoundaryEdgeView {
    std::span<const uint32_t> Blocks; // The mesh's owned edge blocks in ascending order.
    std::span<const SelectionAggregate> Aggregates;
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
        for (const auto block : Blocks) {
            if (!(Aggregates[block].Flags & SelectionBoundary)) continue;
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                for (auto bits = Membership[block].Live[w]; bits; bits &= bits - 1u) {
                    const auto edge = block * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
                    const auto h = EdgeHalfedges[edge];
                    if (h != InvalidOffset && Opposites[h] == InvalidOffset) visit(edge);
                }
        }
    }
    uint32_t Count() const {
        uint32_t count = 0u;
        ForEach([&](uint32_t) { ++count; });
        return count;
    }
};
