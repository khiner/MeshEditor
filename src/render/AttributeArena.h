#pragma once
#include "gpu/BindlessBindings.h"
#include "metal/BufferArena.h"
#include <algorithm>
#include <array>
#include <stdexcept>

// Sparse radix addressing shares allocation and pruning while each attribute
// retains its shader-visible node and payload layout. Zero denotes absence.
template<typename Node, typename T, uint32_t PayloadCount, uint32_t RadixBits, uint32_t Levels>
class AttributeArena {
public:
    explicit AttributeArena(mtl::BufferContext &ctx) : Nodes(ctx, SlotType::Buffer), Values(ctx, SlotType::Buffer) {}
    BufferArena<Node> Nodes;
    BufferArena<std::array<T, PayloadCount>> Values;

protected:
    uint32_t NewNode() {
        auto allocation = Nodes.BeginAllocation();
        const auto node = Nodes.Allocate(1u);
        Nodes.GetMutable(node)[0] = {};
        allocation.Commit();
        return node.Offset;
    }
    uint32_t Attach(uint32_t &root, uint32_t block) {
        if (block >= (1u << (RadixBits * Levels))) throw std::out_of_range("Attribute block exceeds canonical address space.");
        if (root == InvalidOffset) root = NewNode();
        auto id = root;
        for (uint32_t level = Levels; level--;) {
            const auto slot = (block >> (level * RadixBits)) & ((1u << RadixBits) - 1u);
            auto child = Nodes.Get({id, 1u})[0].Children[slot];
            if (!child) {
                if (level) {
                    const auto next = NewNode();
                    try {
                        Nodes.GetMutable({id, 1u})[0].Children[slot] = next + 1u;
                    } catch (...) {
                        Nodes.Release({next, 1u});
                        throw;
                    }
                    child = next + 1u;
                } else {
                    auto allocation = Values.BeginAllocation();
                    const auto payload = Values.Allocate(1u);
                    if (uint64_t(payload.Offset) * PayloadCount > UINT32_MAX) throw std::length_error("Attribute values exceed canonical address space.");
                    child = payload.Offset + 1u;
                    Nodes.GetMutable({id, 1u})[0].Children[slot] = child;
                    allocation.Commit();
                }
            }
            id = child - 1u;
        }
        return id;
    }
    void Detach(uint32_t &root, uint32_t block, bool keep_root = false) {
        if (root == InvalidOffset) return;
        std::array<uint32_t, Levels> path{}, slots{};
        auto id = root;
        for (uint32_t depth = 0u; depth < Levels; ++depth) {
            path[depth] = id;
            slots[depth] = (block >> ((Levels - 1u - depth) * RadixBits)) & ((1u << RadixBits) - 1u);
            const auto child = Nodes.Get({id, 1u})[0].Children[slots[depth]];
            if (!child) return;
            id = child - 1u;
        }
        Values.Release({id, 1u});
        for (uint32_t depth = Levels; depth--;) {
            Nodes.GetMutable({path[depth], 1u})[0].Children[slots[depth]] = 0u;
            if ((!depth && keep_root) || std::ranges::any_of(Nodes.Get({path[depth], 1u})[0].Children, [](auto child) { return child != 0u; })) return;
            Nodes.Release({path[depth], 1u});
        }
        root = InvalidOffset;
    }
    void Release(uint32_t &root) {
        if (root == InvalidOffset) return;
        const auto visit = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            for (const auto child : Nodes.Get({id, 1u})[0].Children)
                if (child) {
                    if (level) self(self, child - 1u, level - 1u);
                    else Values.Release({child - 1u, 1u});
                }
            Nodes.Release({id, 1u});
        };
        visit(visit, root, Levels - 1u);
        root = InvalidOffset;
    }
};
