#pragma once

#include "gpu/Types.h"

#ifndef __METAL_VERSION__
#include <compare>
#include <span>
#endif

// Element domains (vertices, halfedges, edges, faces and triangles) hold 256 slots per block.
GPU_CONSTANT uint32_t MeshElementBlockSize = 256u;
GPU_CONSTANT uint32_t MeshElementBlockWords = MeshElementBlockSize / 32u;
// Render domains (clusters, cluster groups, LOD nodes, LOD members and primitives) hold few members per record, in small blocks.
GPU_CONSTANT uint32_t RenderBlockSize = 8u;

// One block of a domain's arena: its live mask and its owner set, whose table names it at entry Local.
// A record-relative handle is Local times the domain's block size plus the slot, and a block's live mask uses one bit per slot.
struct MeshElementBlock {
    uint32_t Owner DEFAULT(InvalidOffset);
    uint32_t Local DEFAULT(InvalidOffset);
    uint32_t Count DEFAULT();
    GpuArray<uint32_t, MeshElementBlockWords> Live DEFAULT();
};
static_assert(RenderBlockSize <= 32u, "A render block's live mask is one word");

#ifdef __METAL_VERSION__
// The arena address of a record-relative handle of a domain with `BlockSize`-slot blocks, through `table`, the first entry of its record's table run.
template<uint BlockSize, typename Table> inline uint Resolve(Table table, uint handle) {
    return table[handle / BlockSize] * BlockSize + handle % BlockSize;
}
#else
// A record-relative element handle, its local block times the block size plus its slot, meaningful only with its record's table.
struct ElementHandle {
    uint32_t Value{InvalidOffset};
    auto operator<=>(const ElementHandle &) const = default;
};
// An element's slot in its domain's arena, the identity that travels without its record.
// A table resolves a handle to one, a block names one at each of its slots, and a word a buffer or list stores holds one a table resolved.
// A default address names no element.
class ElementAddress {
public:
    ElementAddress() = default;
    template<uint32_t BlockSize> static ElementAddress InBlock(uint32_t block, uint32_t slot) { return ElementAddress{block * BlockSize + slot}; }
    static ElementAddress Stored(uint32_t value) { return ElementAddress{value}; }
    uint32_t Value() const { return Raw; }
    // The address `offset` slots on, within a run of consecutive slots.
    ElementAddress operator+(uint32_t offset) const { return ElementAddress{Raw + offset}; }
    auto operator<=>(const ElementAddress &) const = default;

private:
    explicit ElementAddress(uint32_t value) : Raw(value) {}
    uint32_t Raw{InvalidOffset};
};
// The arena address of a handle of a domain with `BlockSize`-slot blocks, through `table`, its record's table run.
template<uint32_t BlockSize> ElementAddress Resolve(std::span<const uint32_t> table, ElementHandle handle) {
    return ElementAddress::InBlock<BlockSize>(table[handle.Value / BlockSize], handle.Value % BlockSize);
}
#endif
