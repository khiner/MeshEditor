#pragma once

#include "mesh/SelectionIndex.h"

// The blocks the tree marks as `kind`, in ascending local order.
inline std::vector<uint32_t> MarkedBlocks(const SelectionIndexView &tree, SelectionIndexMask kind, std::span<const MeshElementBlock> membership) {
    std::vector<uint32_t> blocks;
    tree.Visit(kind, [&](uint32_t block) { blocks.push_back(block); });
    const auto local = [&](uint32_t block) { return membership[block].Local; };
    if (!std::ranges::is_sorted(blocks, {}, local)) std::ranges::sort(blocks, {}, local);
    return blocks;
}

struct SelectionView {
    std::span<const uint32_t> Bits; // The domain's mask words, by arena address
    std::span<const MeshElementBlock> Membership;
    std::span<const uint32_t> Table;
    SelectionIndexView Tree;
    uint32_t Selected{};
    SelectionIndexMask Kind{SelectionIndexMask::Selected};

    uint32_t Count() const { return Selected; }
    bool Contains(uint32_t handle) const {
        if (handle / MeshElementBlockSize >= Table.size() || Table[handle / MeshElementBlockSize] == InvalidOffset) return false;
        const auto address = Resolve<MeshElementBlockSize>(Table, ElementHandle{handle}).Value();
        return address / 32u < Bits.size() && (Bits[address / 32u] & (1u << (address % 32u))) != 0u;
    }
    // Visits each block holding a marked element in ascending local order, with the number of marked elements before it, and returns their total.
    uint32_t ForEachBlock(auto &&visit) const {
        uint32_t before = 0u;
        if (!Selected) return before;
        for (const auto block : MarkedBlocks(Tree, Kind, Membership)) {
            visit(block, before);
            before += Kind == SelectionIndexMask::Hidden ? Tree.Leaves[block].Hidden : Tree.Leaves[block].Selected;
        }
        return before;
    }
    void ForEach(auto &&visit) const {
        ForEachBlock([&](uint32_t block, uint32_t) {
            const auto base = Membership[block].Local * MeshElementBlockSize;
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                for (auto bits = Bits[block * MeshElementBlockWords + w]; bits; bits &= bits - 1u) {
                    visit(base + w * 32u + uint32_t(std::countr_zero(bits)));
                }
            }
        });
    }
};
