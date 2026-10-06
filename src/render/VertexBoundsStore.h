#pragma once
#include "gpu/AABB.h"
#include "gpu/BindlessBindings.h"
#include "gpu/VertexBounds.h"
#include "metal/Buffer.h"
#include "metal/PhysicalPages.h"
#include "render/AttributeArena.h"
#include "state/Entity.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Canonical bounds keys address one packed value arena.
// Three small radix levels map 32-value blocks.
// Exact membership excludes stale holes.
class VertexBoundsStore : public AttributeArena<VertexBoundsMapNode, AABB, 32u, 7u, 3u> {
    using Arena = AttributeArena<VertexBoundsMapNode, AABB, 32u, 7u, 3u>;

public:
    explicit VertexBoundsStore(mtl::BufferContext &ctx) : Arena(ctx), Members(ctx, 0u, SlotType::Buffer) {}
    using Keys = std::array<std::vector<uint32_t>, VertexBoundsLevels>;
    struct Prepared {
        std::span<const uint32_t> Roots;
        const Keys &Nodes;
        bool Changed;
        uint64_t LayoutRevision;
    };
    struct ReadView {
        std::span<const VertexBoundsMapNode> Nodes;
        std::span<const AABB> Values;
        std::span<const uint32_t> Members;
        uint32_t Root;
        uint32_t Mapped(uint32_t key) const {
            if (Root == InvalidOffset || key >= (1u << 26u)) return InvalidOffset;
            uint32_t id = Root;
            for (uint32_t level = 3u; level--;) {
                const auto child = Nodes[id].Children[(key >> (5u + level * 7u)) & 127u];
                if (!child) return InvalidOffset;
                id = child - 1u;
            }
            return id * 32u + key % 32u;
        }
        uint32_t Find(uint32_t key) const {
            const auto at = Mapped(key);
            return at != InvalidOffset && (Members[at / 32u] & (1u << (key % 32u))) ? at : InvalidOffset;
        }
        uint32_t Index(uint32_t key) const {
            const auto at = Find(key);
            assert(at != InvalidOffset);
            return at;
        }
        const AABB &operator[](uint32_t key) const { return Values[Index(key)]; }
    };
    ReadView View(uint32_t root) const { return {Nodes.Buffer.GetSpan<VertexBoundsMapNode>(), Values.Buffer.GetSpan<AABB>(), Members.GetSpan<uint32_t>(), root}; }
    void BeginUpdate() { ++Epoch; }
    template<typename GetBlocks>
    Prepared Prepare(state::Entity entity, uint32_t store, uint64_t revision, uint32_t count, GetBlocks &&get_blocks) {
        auto &group = Groups[entity];
        group.Seen = Epoch;
        if (group.Store == store && group.Revision == revision && group.Roots.size() == count) return {group.Roots, group.Keys, false, group.LayoutRevision};
        if (group.Store != store) {
            Unregister(group.Store, entity);
            ByStore[store].insert(entity);
        }
        const auto blocks = get_blocks();
        for (const auto block : blocks)
            if (block >= (1u << 24u)) throw std::out_of_range("Vertex bounds block exceeds canonical address space.");
        const std::unordered_set<uint32_t> membership(blocks.begin(), blocks.end());
        bool changed = group.Store != store || group.Roots.size() != count;
        while (group.Roots.size() > count) {
            Release(group.Roots.back());
            group.Roots.pop_back();
        }
        std::unordered_set<uint32_t> changed_masks;
        // Removal swaps the last key into its slot, so inspect that slot again.
        for (size_t i = 0u; i < group.Keys[0].size();) {
            const auto block = group.Keys[0][i];
            if (membership.contains(block)) ++i;
            else changed |= SetBlock(group, block, false, &changed_masks);
        }
        for (const auto block : blocks) changed |= SetBlock(group, block, true, &changed_masks);
        if (group.Keys.back().empty()) AddKey(group, VertexBoundsLevels - 1u, 0u, &changed_masks);
        for (const auto block : changed_masks) PublishMask(group, block);
        while (group.Roots.size() < count) {
            const auto root = NewNode();
            group.Roots.push_back(root);
            for (const auto [block, mask] : group.Masks) SetMask(root, block, mask);
        }
        if (changed) ++group.LayoutRevision;
        group.Store = store;
        group.Revision = revision;
        return {group.Roots, group.Keys, true, group.LayoutRevision};
    }
    // A canonical edit publishes only changed vertex blocks. Keep the pose
    // bounds dispatch keys and their sparse address map at that revision.
    template<typename Present>
    void UpdateBlocks(uint32_t store, uint64_t revision, std::span<const uint32_t> blocks, Present &&present) {
        const auto found = ByStore.find(store);
        if (found == ByStore.end()) return;
        for (const auto entity : found->second) {
            auto &group = Groups.at(entity);
            for (const auto block : blocks)
                if (SetBlock(group, block, present(block))) ++group.LayoutRevision;
            group.Revision = revision;
        }
    }
    void EndUpdate() {
        for (auto it = Groups.begin(); it != Groups.end();) {
            if (it->second.Seen == Epoch) {
                ++it;
                continue;
            }
            for (auto &root : it->second.Roots) Release(root);
            Unregister(it->second.Store, it->first);
            it = Groups.erase(it);
        }
    }
    void Reset() {
        ByStore.clear();
        Groups.clear();
        Nodes.Reset();
        Values.Reset();
        Members.SetUsedSize(0u);
        Epoch = 0u;
    }
    mtl::Buffer Members;

private:
    struct Group {
        uint32_t Store{InvalidOffset};
        uint64_t Revision{}, Seen{}, LayoutRevision{};
        VertexBoundsStore::Keys Keys;
        std::array<std::unordered_map<uint32_t, uint32_t>, VertexBoundsLevels> Positions;
        std::array<std::unordered_map<uint32_t, uint32_t>, VertexBoundsLevels - 1u> Children;
        std::unordered_map<uint32_t, uint32_t> Masks;
        std::vector<uint32_t> Roots;
    };
    std::unordered_map<state::Entity, Group> Groups;
    std::unordered_map<uint32_t, std::unordered_set<state::Entity>> ByStore;
    uint64_t Epoch{};
    void Unregister(uint32_t store, state::Entity entity) {
        const auto found = ByStore.find(store);
        if (found == ByStore.end()) return;
        found->second.erase(entity);
        if (found->second.empty()) ByStore.erase(found);
    }
    bool SetBlock(Group &group, uint32_t block, bool present, std::unordered_set<uint32_t> *changed_masks = nullptr) {
        if (group.Positions[0].contains(block) == present) return false;
        if (present) AddKey(group, 0u, block, changed_masks);
        else RemoveKey(group, 0u, block, changed_masks);
        for (uint32_t level = 1u; level + 1u < VertexBoundsLevels; ++level) {
            block /= 256u;
            if (present) {
                if (++group.Children[level][block] != 1u) break;
                AddKey(group, level, block, changed_masks);
            } else {
                auto it = group.Children[level].find(block);
                if (it == group.Children[level].end() || !it->second) throw std::logic_error("Vertex bounds parent count is missing.");
                if (--it->second) break;
                group.Children[level].erase(it);
                RemoveKey(group, level, block, changed_masks);
            }
        }
        return true;
    }
    void SetMask(uint32_t root, uint32_t block, uint32_t mask) {
        const auto payload = Attach(root, block);
        Members.SetUsedSize(uint64_t(Values.Buffer.Count<std::array<AABB, 32>>()) * sizeof(uint32_t));
        Members.GetMutableSpan<uint32_t>({payload, 1u})[0] = mask;
    }
    void PublishMask(Group &group, uint32_t block, std::unordered_set<uint32_t> *changed_masks = nullptr) {
        if (changed_masks) {
            changed_masks->insert(block);
            return;
        }
        const auto mask = group.Masks.find(block);
        for (auto root : group.Roots) {
            if (mask == group.Masks.end()) Detach(root, block, true);
            else SetMask(root, block, mask->second);
        }
    }
    void AddKey(Group &group, uint32_t level, uint32_t key, std::unordered_set<uint32_t> *changed_masks = nullptr) {
        auto &keys = group.Keys[level];
        if (!group.Positions[level].emplace(key, uint32_t(keys.size())).second) return;
        keys.push_back(key);
        const auto record = VertexBoundsKey(level, key), block = record / 32u;
        const auto mask = group.Masks[block] | (1u << (record % 32u));
        group.Masks[block] = mask;
        PublishMask(group, block, changed_masks);
    }
    void RemoveKey(Group &group, uint32_t level, uint32_t key, std::unordered_set<uint32_t> *changed_masks = nullptr) {
        auto &keys = group.Keys[level];
        const auto position = group.Positions[level].at(key);
        keys[position] = keys.back();
        group.Positions[level][keys.back()] = position;
        keys.pop_back();
        group.Positions[level].erase(key);
        const auto record = VertexBoundsKey(level, key), block = record / 32u;
        auto &mask = group.Masks.at(block);
        mask &= ~(1u << (record % 32u));
        if (!mask) group.Masks.erase(block);
        PublishMask(group, block, changed_masks);
    }
};
