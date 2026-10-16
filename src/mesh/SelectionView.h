#pragma once

#include "mesh/SelectionIndex.h"

// The blocks the tree marks as `kind`, in ascending local order.
inline std::vector<uint32_t> MarkedBlocks(const SelectionIndexView &tree, SelectionIndexMask kind, std::span<const MeshElementBlock> membership) {
    std::vector<uint32_t> blocks;
    tree.Visit(kind, false, [&](uint32_t block) {
        blocks.push_back(block);
        return true;
    });
    const auto local = [&](uint32_t block) { return membership[block].Local; };
    if (!std::ranges::is_sorted(blocks, {}, local)) std::ranges::sort(blocks, {}, local);
    return blocks;
}

// Borrowed masks over one mesh domain's owned blocks, visited in ascending handle order.
struct SelectionView {
    std::span<const uint32_t> Bits; // The domain's mask words, by arena address
    std::span<const MeshElementBlock> Membership; // The domain's block metadata
    std::span<const uint32_t> Table; // The record's table entries in the domain
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
    // Visits the arena address of each marked element, in ascending block order.
    void ForEachAddress(auto &&visit) const {
        if (!Selected) return;
        Tree.Visit(Kind, false, [&](uint32_t block) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                for (auto bits = Bits[block * MeshElementBlockWords + w]; bits; bits &= bits - 1u) {
                    visit(ElementAddress::InBlock<MeshElementBlockSize>(block, w * 32u + uint32_t(std::countr_zero(bits))));
                }
            }
            return true;
        });
    }
    std::optional<uint32_t> Extreme(bool last) const {
        if (!Selected) return {};
        const auto blocks = MarkedBlocks(Tree, Kind, Membership);
        if (blocks.empty()) return {};
        const auto block = last ? blocks.back() : blocks.front();
        for (uint32_t i = 0u; i < MeshElementBlockWords; ++i) {
            const auto w = last ? MeshElementBlockWords - 1u - i : i;
            if (const auto bits = Bits[block * MeshElementBlockWords + w]) {
                return Membership[block].Local * MeshElementBlockSize + w * 32u + (last ? 31u - uint32_t(std::countl_zero(bits)) : uint32_t(std::countr_zero(bits)));
            }
        }
        return {};
    }
    std::optional<uint32_t> First() const { return Extreme(false); }
    std::optional<uint32_t> Last() const { return Extreme(true); }
};

// The live edges of one mesh without an opposite halfedge, in ascending handle order.
// Only edge blocks whose aggregate carries the boundary flag are visited.
struct BoundaryEdgeView {
    SelectionIndexView Tree;
    std::span<const MeshElementBlock> Membership;
    std::span<const uint32_t> EdgeTable, HalfedgeTable; // The record's table entries
    std::span<const uint32_t> EdgeHalfedges, Opposites;

    bool Boundary(ElementAddress address) const {
        const auto h = EdgeHalfedges[address.Value()];
        return h != InvalidOffset && Opposites[Resolve<MeshElementBlockSize>(HalfedgeTable, ElementHandle{h}).Value()] == InvalidOffset;
    }
    bool Contains(uint32_t edge) const {
        if (edge / MeshElementBlockSize >= EdgeTable.size() || EdgeTable[edge / MeshElementBlockSize] == InvalidOffset) return false;
        const auto address = Resolve<MeshElementBlockSize>(EdgeTable, ElementHandle{edge});
        const auto slot = edge % MeshElementBlockSize;
        return (Membership[address.Value() / MeshElementBlockSize].Live[slot / 32u] >> (slot % 32u) & 1u) && Boundary(address);
    }
    void ForEach(auto &&visit) const {
        for (const auto block : MarkedBlocks(Tree, SelectionIndexMask::Boundary, Membership)) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                for (auto bits = Membership[block].Live[w]; bits; bits &= bits - 1u) {
                    const auto slot = w * 32u + uint32_t(std::countr_zero(bits));
                    if (Boundary(ElementAddress::InBlock<MeshElementBlockSize>(block, slot))) visit(Membership[block].Local * MeshElementBlockSize + slot);
                }
            }
        }
    }
    uint32_t Count() const {
        uint32_t count = 0u;
        ForEach([&](uint32_t) { ++count; });
        return count;
    }
};
