#pragma once
#include "gpu/BindlessBindings.h"
#include "gpu/PoseAttributeNode.h"
#include "mesh/PoseAttributeView.h"
#include "metal/Buffer.h"
#include "metal/PhysicalPages.h"
#include "render/AttributeArena.h"
#include "state/Entity.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Typed payloads share one node arena and one value arena per attribute.
// Namespace roots isolate poses while canonical keys retain stable addresses.
template<typename T>
class PoseAttributeStore : public AttributeArena<PoseAttributeNode, T, 256u, PoseAttributeRadixBits, 2u> {
    using Arena = AttributeArena<PoseAttributeNode, T, 256u, PoseAttributeRadixBits, 2u>;
    using Arena::Attach;
    using Arena::Detach;
    using Arena::Release;

public:
    explicit PoseAttributeStore(mtl::BufferContext &ctx) : Arena(ctx) {}
    using Arena::Nodes;
    using Arena::Values;
    void BeginUpdate() { ++Epoch; }
    struct Prepared {
        std::span<const uint32_t> Roots;
        bool Changed;
        uint64_t LayoutRevision;
    };
    template<typename GetBlocks>
    Prepared Prepare(state::Entity entity, uint32_t store, uint64_t revision, uint32_t count, GetBlocks &&get_blocks, uint32_t domain = 0u) {
        auto &group = Groups[entity];
        group.Seen = Epoch;
        if (group.Store == store && group.Domain == domain && group.Revision == revision && group.Roots.size() == count) {
            return {group.Roots, false, group.LayoutRevision};
        }
        auto blocks = get_blocks();
        std::unordered_set<uint32_t> membership(blocks.begin(), blocks.end());
        if (group.Store != store || group.Domain != domain || group.Roots.size() != count || group.Blocks != membership) ++group.LayoutRevision;
        if (group.Store != store) {
            Unregister(group.Store, entity);
            ByStore[store].insert(entity);
        }
        for (const auto block : blocks)
            if (block >= (1u << 24u)) throw std::length_error("Pose attribute block exceeds canonical address space.");
        while (group.Roots.size() > count) {
            Release(group.Roots.back());
            group.Roots.pop_back();
        }
        for (auto &root : group.Roots) {
            for (const auto old : group.Blocks)
                if (!membership.contains(old)) Detach(root, old);
            for (const auto block : blocks)
                if (!group.Blocks.contains(block)) Attach(root, block);
        }
        while (group.Roots.size() < count) {
            group.Roots.push_back(InvalidOffset);
            for (const auto block : blocks) Attach(group.Roots.back(), block);
        }
        group.Store = store;
        group.Domain = domain;
        group.Revision = revision;
        group.Blocks = std::move(membership);
        return {group.Roots, true, group.LayoutRevision};
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
    PoseAttributeView<T> View(uint32_t root) const {
        if (root == InvalidOffset) return {};
        return {Nodes.Buffer.template GetSpan<PoseAttributeNode>(), Values.Buffer.template GetSpan<T>(), root};
    }
    void Reset() {
        ByStore.clear();
        Groups.clear();
        Nodes.Reset();
        Values.Reset();
        Epoch = 0;
    }

    // Publication supplies changed canonical block IDs, so unchanged groups
    // adopt the revision without re-enumerating their surviving membership.
    template<typename Present>
    void UpdateBlocks(uint32_t store, uint64_t revision, std::span<const uint32_t> blocks, Present &&present, uint32_t domain = 0u) {
        const auto found = ByStore.find(store);
        if (found == ByStore.end()) return;
        for (const auto entity : found->second) {
            auto &group = Groups.at(entity);
            if (group.Domain != domain) continue;
            bool changed = false;
            for (const auto block : blocks) {
                if (present(block)) {
                    if (group.Blocks.insert(block).second) {
                        changed = true;
                        for (auto &root : group.Roots) Attach(root, block);
                    }
                } else if (group.Blocks.erase(block)) {
                    changed = true;
                    for (auto &root : group.Roots) Detach(root, block);
                }
            }
            if (changed) ++group.LayoutRevision;
            group.Revision = revision;
        }
    }
    // Temporary morph probes use the same storage and addressing as live poses.
    struct Temporary {
        explicit Temporary(PoseAttributeStore &store) : Store(store) {}
        Temporary(const Temporary &) = delete;
        ~Temporary() {
            for (auto &root : Roots) Store.Release(root);
        }
        uint32_t Add(std::span<const uint32_t> blocks) {
            Roots.push_back(InvalidOffset);
            for (const auto block : blocks) Store.Attach(Roots.back(), block);
            return Roots.back();
        }
        PoseAttributeStore &Store;
        std::vector<uint32_t> Roots;
    };

private:
    struct Group {
        uint32_t Store{InvalidOffset};
        uint32_t Domain{};
        uint64_t Revision{}, Seen{}, LayoutRevision{};
        std::unordered_set<uint32_t> Blocks;
        std::vector<uint32_t> Roots;
    };
    std::unordered_map<state::Entity, Group> Groups;
    std::unordered_map<uint32_t, std::unordered_set<state::Entity>> ByStore;
    uint64_t Epoch{};
    void Unregister(uint32_t store, state::Entity entity) {
        const auto it = ByStore.find(store);
        if (it == ByStore.end()) return;
        it->second.erase(entity);
        if (it->second.empty()) ByStore.erase(it);
    }
};
