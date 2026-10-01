#pragma once
#include "Range.h"
#include "gpu/BindlessBindings.h"
#include "gpu/MeshletIndex.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"
#include <bit>

struct MeshletIndexEdit {
    uint32_t Root{InvalidOffset};
    // Canonical handles in any order.
    std::span<const uint32_t> Added{}, Removed{};
    Range Insert{};
};

struct MeshletIndex {
    explicit MeshletIndex(mtl::BufferContext &ctx) : Nodes(ctx,SlotType::Buffer), Leaves(ctx,SlotType::Buffer) {}
    MeshletIndexRef Ref(uint32_t root) const { return {Nodes.Buffer.Slot,Leaves.Buffer.Slot,root}; }
    uint32_t Count(uint32_t root) const { return root == InvalidOffset ? 0u : Nodes.Get({root,1u})[0].Count; }
    bool Contains(uint32_t root, uint32_t element) const {
        if (root == InvalidOffset) return false;
        auto id = root;
        const auto block = element / 256u;
        for (uint32_t level = MeshletIndexLevels; level--;) {
            const auto &node = Nodes.Get({id,1u})[0];
            const auto slot = (block >> (level*5u))&31u;
            if (!(node.Active & (1u<<slot))) return false;
            id = node.Children[slot];
        }
        return (Leaves.Get({id,1u})[0].Live[(element%256u)/32u] & (1u<<(element%32u))) != 0u;
    }
    bool HasBlock(uint32_t root, uint32_t block) const {
        if (root == InvalidOffset) return false;
        auto id = root;
        for (uint32_t level = MeshletIndexLevels; level--;) {
            const auto &node = Nodes.Get({id,1u})[0];
            const auto slot = (block >> (level*5u))&31u;
            if (!(node.Active & (1u<<slot))) return false;
            id = node.Children[slot];
        }
        return Leaves.Get({id,1u})[0].Count != 0u;
    }
    void ForEachBlock(uint32_t root, auto &&fn) const {
        if (!Count(root)) return;
        const auto visit = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            const auto &node = Nodes.Get({id,1u})[0];
            for (auto active = node.Active; active; active &= active-1u) {
                const auto child = node.Children[std::countr_zero(active)];
                if (level) self(self,child,level-1u);
                else fn(Leaves.Get({child,1u})[0].Block);
            }
        };
        visit(visit,root,MeshletIndexLevels-1u);
    }
    uint32_t First(uint32_t root) const {
        if (!Count(root)) return InvalidOffset;
        auto id = root;
        for (uint32_t level = 0u; level < MeshletIndexLevels; ++level) {
            const auto &node = Nodes.Get({id,1u})[0];
            id = node.Children[std::countr_zero(node.Active)];
        }
        const auto &leaf = Leaves.Get({id,1u})[0];
        for (uint32_t w = 0u; w < 8u; ++w)
            if (leaf.Live[w]) return leaf.Block*256u+w*32u+uint32_t(std::countr_zero(leaf.Live[w]));
        return InvalidOffset;
    }
    void ForEach(uint32_t root, auto &&fn) const {
        if (!Count(root)) return;
        const auto visit = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            const auto &node = Nodes.Get({id,1u})[0];
            for (auto active = node.Active; active; active &= active-1u) {
                const auto child = node.Children[std::countr_zero(active)];
                if (level) self(self,child,level-1u);
                else {
                    const auto &leaf = Leaves.Get({child,1u})[0];
                    for (uint32_t w = 0u; w < 8u; ++w)
                        for (auto bits = leaf.Live[w]; bits; bits &= bits-1u)
                            fn(leaf.Block*256u+w*32u+uint32_t(std::countr_zero(bits)));
                }
            }
        };
        visit(visit,root,MeshletIndexLevels-1u);
    }
    // Each root occurs once per batch.
    // Removal precedes insertion.
    // Duplicate insertions and missing removals are idempotent.
    // Empty roots remain valid.
    // The host rewrites only the leaves and nodes on the edited blocks' paths.
    void Update(std::span<MeshletIndexEdit>);
    void Release(uint32_t root);
    void Reset() { Nodes.Reset(); Leaves.Reset(); }
    BufferArena<MeshletIndexNode> Nodes;
    BufferArena<MeshletIndexLeaf> Leaves;
};
