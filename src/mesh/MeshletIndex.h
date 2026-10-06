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

struct MeshletIndexView {
    std::span<const MeshletIndexNode> Nodes;
    std::span<const MeshletIndexLeaf> Leaves;
    // Stream roots by level in batches that keep short trees together in cache.
    // Callbacks receive the root's input index.
    void Visit(std::span<const uint32_t> roots, auto &&node_fn, auto &&leaf_fn) const {
        struct Entry {
            uint32_t Root, Id;
        };
        std::vector<Entry> current, next;
        constexpr size_t batch_size = 4096u;
        current.reserve(std::min(roots.size(), batch_size));
        for (size_t first = 0u; first < roots.size(); first += batch_size) {
            current.clear();
            const auto end = std::min(first + batch_size, roots.size());
            for (size_t i = first; i < end; ++i)
                if (roots[i] != InvalidOffset) current.push_back({uint32_t(i), roots[i]});
            for (uint32_t level = MeshletIndexLevels; level-- && !current.empty();) {
                next.clear();
                for (const auto [root, id] : current) {
                    const auto &node = Nodes[id];
                    node_fn(root, id);
                    for (auto active = node.Active; active; active &= active - 1u) {
                        const auto child = node.Children[std::countr_zero(active)];
                        if (level) next.push_back({root, child});
                        else leaf_fn(root, child, Leaves[child]);
                    }
                }
                current.swap(next);
            }
        }
    }
    // Compile allocation runs while callers collect each leaf's owned payloads.
    void CollectOwned(std::span<const uint32_t> roots, std::vector<Range> &nodes, std::vector<Range> &leaves, auto &&leaf_fn) const {
        std::vector<Range> pending(roots.size());
        Visit(roots, [&](uint32_t root, uint32_t id) {
            auto &run = pending[root];
            if (run.Count && run.Offset + run.Count == id) ++run.Count;
            else if (run.Count && id + 1u == run.Offset) { --run.Offset; ++run.Count; }
            else { AppendRange(nodes, run); run = {id, 1u}; } }, [&](uint32_t root, uint32_t id, const MeshletIndexLeaf &leaf) {
            AppendRange(leaves, {id, 1u});
            leaf_fn(root, leaf); });
        for (const auto run : pending) AppendRange(nodes, run);
    }
    // Visit each owned node and leaf once, including an empty root.
    void Visit(uint32_t root, auto &&node_fn, auto &&leaf_fn) const {
        if (root == InvalidOffset) return;
        const auto visit = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            const auto &node = Nodes[id];
            node_fn(id);
            for (auto active = node.Active; active; active &= active - 1u) {
                const auto child = node.Children[std::countr_zero(active)];
                if (level) self(self, child, level - 1u);
                else leaf_fn(child, Leaves[child]);
            }
        };
        visit(visit, root, MeshletIndexLevels - 1u);
    }
};

struct MeshletIndex {
    explicit MeshletIndex(mtl::BufferContext &ctx) : Nodes(ctx, SlotType::Buffer), Leaves(ctx, SlotType::Buffer) {}
    MeshletIndexView Read() const { return {Nodes.Buffer.GetSpan<MeshletIndexNode>(), Leaves.Buffer.GetSpan<MeshletIndexLeaf>()}; }
    MeshletIndexRef Ref(uint32_t root) const { return {Nodes.Buffer.Slot, Leaves.Buffer.Slot, root}; }
    uint32_t Count(uint32_t root) const { return root == InvalidOffset ? 0u : Nodes.Get({root, 1u})[0].Count; }
    bool Contains(uint32_t root, uint32_t element) const {
        if (root == InvalidOffset) return false;
        auto id = root;
        const auto block = element / 256u;
        for (uint32_t level = MeshletIndexLevels; level--;) {
            const auto &node = Nodes.Get({id, 1u})[0];
            const auto slot = (block >> (level * 5u)) & 31u;
            if (!(node.Active & (1u << slot))) return false;
            id = node.Children[slot];
        }
        return (Leaves.Get({id, 1u})[0].Live[(element % 256u) / 32u] & (1u << (element % 32u))) != 0u;
    }
    bool HasBlock(uint32_t root, uint32_t block) const {
        if (root == InvalidOffset) return false;
        auto id = root;
        for (uint32_t level = MeshletIndexLevels; level--;) {
            const auto &node = Nodes.Get({id, 1u})[0];
            const auto slot = (block >> (level * 5u)) & 31u;
            if (!(node.Active & (1u << slot))) return false;
            id = node.Children[slot];
        }
        return Leaves.Get({id, 1u})[0].Count != 0u;
    }
    void ForEachBlock(uint32_t root, auto &&fn) const {
        Visit(root, [](uint32_t) {}, [&](uint32_t, const MeshletIndexLeaf &leaf) { fn(leaf.Block); });
    }
    void Visit(uint32_t root, auto &&node_fn, auto &&leaf_fn) const {
        Read().Visit(root, node_fn, leaf_fn);
    }
    uint32_t First(uint32_t root) const {
        if (!Count(root)) return InvalidOffset;
        auto id = root;
        for (uint32_t level = 0u; level < MeshletIndexLevels; ++level) {
            const auto &node = Nodes.Get({id, 1u})[0];
            id = node.Children[std::countr_zero(node.Active)];
        }
        const auto &leaf = Leaves.Get({id, 1u})[0];
        for (uint32_t w = 0u; w < 8u; ++w)
            if (leaf.Live[w]) return leaf.Block * 256u + w * 32u + uint32_t(std::countr_zero(leaf.Live[w]));
        return InvalidOffset;
    }
    void ForEach(uint32_t root, auto &&fn) const {
        Visit(root, [](uint32_t) {}, [&](uint32_t, const MeshletIndexLeaf &leaf) { ForEach(leaf, fn); });
    }
    static void ForEach(const MeshletIndexLeaf &leaf, auto &&fn) {
        for (uint32_t w = 0u; w < 8u; ++w)
            for (auto bits = leaf.Live[w]; bits; bits &= bits - 1u)
                fn(leaf.Block * 256u + w * 32u + uint32_t(std::countr_zero(bits)));
    }
    // Each root occurs once per batch.
    // Removal precedes insertion.
    // Duplicate insertions and missing removals are idempotent.
    // Empty roots remain valid.
    // The host rewrites only the leaves and nodes on the edited blocks' paths.
    void Update(std::span<MeshletIndexEdit>);
    void Release(uint32_t root);
    void Release(std::span<const uint32_t> roots);
    void Reset() {
        Nodes.Reset();
        Leaves.Reset();
    }
    BufferArena<MeshletIndexNode> Nodes;
    BufferArena<MeshletIndexLeaf> Leaves;
};
