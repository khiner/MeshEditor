#pragma once

#include "gpu/MeshElementBlock.h"
#include "gpu/SelectionAggregate.h"
#include "metal/BufferArena.h"

// Six radix-16 levels address the 24 bits of a canonical 256-element block.
inline constexpr uint32_t SelectionIndexLevels = 6u;
enum class SelectionIndexMask { Selected,
                                Hidden };
struct SelectionIndexNode {
    std::array<uint32_t, 16> Children;
    uint32_t Parent{InvalidOffset}, Slot{}, Active{}, Selected{}, Hidden{};
};

struct SelectionIndexView {
    std::span<const SelectionIndexNode> Nodes;
    std::span<const SelectionAggregate> Leaves;
    uint32_t Root{InvalidOffset};

    void Visit(SelectionIndexMask kind, auto &&visit) const {
        if (Root == InvalidOffset) return;
        const auto walk = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            const auto &node = Nodes[id];
            for (auto mask = kind == SelectionIndexMask::Hidden ? node.Hidden : node.Selected; mask; mask &= mask - 1u) {
                const auto child = node.Children[uint32_t(std::countr_zero(mask))];
                if (level) self(self, child, level - 1u);
                else visit(child);
            }
        };
        walk(walk, Root, SelectionIndexLevels - 1u);
    }
};

// Derived block ownership and partial aggregates over the canonical block aggregates the GPU writes.
struct SelectionIndex {
    explicit SelectionIndex(mtl::BufferContext &ctx) : Aggregates{ctx, SlotType::Buffer} {}
    SelectionIndexView Read(uint32_t root, std::span<const SelectionAggregate> leaves) const {
        return {Nodes, leaves, root < Roots.size() ? Roots[root] : InvalidOffset};
    }
    // Visits each live leaf block whose bounds and ancestors' bounds pass `overlap`.
    // Returns the nodes visited, or zero without a root.
    uint32_t VisitBounds(uint32_t root, std::span<const SelectionAggregate> leaves, auto &&overlap, auto &&visit) const {
        if (root >= Roots.size() || Roots[root] == InvalidOffset) return 0u;
        uint32_t visited = 0u;
        const auto values = Aggregates.Buffer.GetSpan<SelectionAggregate>();
        const auto walk = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            ++visited;
            if (!overlap(values[id].Bounds)) return;
            const auto &node = Nodes[id];
            for (auto mask = node.Active; mask; mask &= mask - 1u) {
                const auto child = node.Children[uint32_t(std::countr_zero(mask))];
                if (level) self(self, child, level - 1u);
                else if (overlap(leaves[child].Bounds)) visit(child);
            }
        };
        walk(walk, Roots[root], SelectionIndexLevels - 1u);
        return visited;
    }
    const SelectionAggregate &Get(uint32_t root) const {
        static const SelectionAggregate empty{};
        return root < Roots.size() && Roots[root] != InvalidOffset ? Aggregates.Get({Roots[root], 1u})[0] : empty;
    }
    SlotOffset Ref(uint32_t root) const {
        return root < Roots.size() && Roots[root] != InvalidOffset ? SlotOffset{Aggregates.Buffer.Slot, Roots[root]} : SlotOffset{};
    }
    void Update(uint32_t root, std::span<const uint32_t> blocks, uint32_t owner, std::span<const MeshElementBlock> membership, std::span<const SelectionAggregate> leaves);
    void Release(uint32_t root);
    uint32_t Owner(uint32_t domain, uint32_t block) const {
        const auto found = Owners[domain].find(block);
        return found == Owners[domain].end() ? InvalidOffset : found->second;
    }
    void Reset() {
        Aggregates.Reset();
        Nodes.clear();
        Roots.clear();
        for (auto &owners : Owners) owners.clear();
    }

    BufferArena<SelectionAggregate> Aggregates;
    std::vector<SelectionIndexNode> Nodes;
    std::vector<uint32_t> Roots;
    // The owner at the last completed update, which lets restore retire a block from its former tree after canonical membership loses its owner.
    std::array<std::unordered_map<uint32_t, uint32_t>, 3> Owners;
};

inline void ReleaseMesh(SelectionIndex &index, uint32_t id) {
    for (uint32_t d = 0u; d < 3u; ++d) index.Release(3u * id + d);
}
