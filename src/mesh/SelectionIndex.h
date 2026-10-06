#pragma once

#include "gpu/MeshElementBlock.h"
#include "gpu/SelectionAggregate.h"
#include "metal/BufferArena.h"

#include <bit>
#include <unordered_map>

// Six radix-16 levels address the 24 bits of a canonical 256-element block.
inline constexpr uint32_t SelectionIndexLevels = 6u;
enum class SelectionIndexMask { Selected,
                                Boundary,
                                Hidden,
                                Unselected };
struct SelectionIndexNode {
    std::array<uint32_t, 16> Children;
    uint32_t Parent{InvalidOffset}, Slot{}, Active{}, Selected{}, Boundary{}, Hidden{}, Unselected{};
};

struct SelectionIndexView {
    std::span<const SelectionIndexNode> Nodes;
    std::span<const SelectionAggregate> Leaves;
    uint32_t Root{InvalidOffset};

    // Only branches containing selected elements (or boundary edges) are visited.
    // A false callback result stops the walk, including for a first/last query.
    void Visit(SelectionIndexMask kind, bool reverse, auto &&visit) const {
        if (Root == InvalidOffset) return;
        const auto walk = [&](auto &&self, uint32_t id, uint32_t level) -> bool {
            const auto &node = Nodes[id];
            auto mask = kind == SelectionIndexMask::Boundary ? node.Boundary : kind == SelectionIndexMask::Hidden ? node.Hidden :
                kind == SelectionIndexMask::Unselected                                                            ? node.Unselected :
                                                                                                                    node.Selected;
            while (mask) {
                const auto slot = reverse ? 31u - uint32_t(std::countl_zero(mask)) : uint32_t(std::countr_zero(mask));
                mask &= ~(1u << slot);
                const auto child = node.Children[slot];
                if (level ? !self(self, child, level - 1u) : !visit(child)) return false;
            }
            return true;
        };
        walk(walk, Root, SelectionIndexLevels - 1u);
    }
};

// Derived ownership and partial aggregates. The GPU produces canonical block
// aggregates; after submission the CPU repairs only their six ancestor paths.
// All GPU-readable aggregate values stay in UMA, including each mesh's root.
struct SelectionIndex {
    explicit SelectionIndex(mtl::BufferContext &ctx) : Aggregates{ctx, SlotType::Buffer} {}
    SelectionIndexView Read(uint32_t root, std::span<const SelectionAggregate> leaves) const {
        return {Nodes, leaves, root < Roots.size() ? Roots[root] : InvalidOffset};
    }
    // Prune canonical block ownership by its existing reduced geometry bounds.
    // Bounds stay in UMA; traversal reads no vertex values and uploads only leaf block IDs.
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
    // The owner at the last completed update. Restore uses it to retire a block
    // from its former tree even when canonical membership now has no owner.
    std::array<std::unordered_map<uint32_t, uint32_t>, 3> Owners;
};
