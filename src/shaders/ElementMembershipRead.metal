#ifndef ELEMENT_MEMBERSHIP_READ_MSL
#define ELEMENT_MEMBERSHIP_READ_MSL

#include "Bindless.metal"
#include "gpu/ElementWorkSeedJob.h"
#include "gpu/MeshElementBlock.h"

// Binary search over a mesh domain's ascending owned blocks, followed by their inclusive live-element prefix.
// A dense mesh ordinal maps to its canonical handle without packing a mesh-sized array.
inline uint LiveElementAt(device const BindlessSet &b, uint blocks_slot, SlotOffset list, uint block_count, uint ordinal) {
    device const uint *blocks = BindlessBuffer(uint,b.Buffer,list.Slot) + list.Offset;
    device const uint *prefix = blocks + block_count;
    uint low = 0u, high = block_count;
    while (low < high) {
        const uint middle = (low + high) / 2u;
        if (prefix[middle] <= ordinal) low = middle + 1u;
        else high = middle;
    }
    if (low >= block_count) return InvalidOffset;
    ordinal -= low ? prefix[low - 1u] : 0u;
    const uint block = blocks[low];
    return SelectLiveElement(&BindlessBuffer(MeshElementBlock,b.Buffer,blocks_slot)[block].Live[0], block, ordinal);
}

// One 256-lane tile reads one owned canonical block. Selection changes only
// which blocks and live bits participate, not geometry addressing.
inline uint MembershipElement(device const BindlessSet &b, ElementWorkSeedJob source, uint group, uint lane) {
    if (group >= source.BlockCount) return InvalidOffset;
    const uint id = BindlessBuffer(uint,b.Buffer,source.BlockIds.Slot)[source.BlockIds.Offset+group];
    const auto block = BindlessBuffer(MeshElementBlock,b.Buffer,source.BlocksSlot)[id];
    if (block.Owner != source.Owner) return InvalidOffset;
    uint live = block.Live[lane/32u];
    if (source.SelectionSlot != InvalidSlot) live &= BindlessBuffer(uint,b.Buffer,source.SelectionSlot)[id*8u+lane/32u];
    return live & (1u<<(lane%32u)) ? id*256u+lane : InvalidOffset;
}

#endif
