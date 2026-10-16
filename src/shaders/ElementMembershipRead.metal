#ifndef ELEMENT_MEMBERSHIP_READ_MSL
#define ELEMENT_MEMBERSHIP_READ_MSL

#include "Bindless.metal"
#include "gpu/ElementWorkSeedJob.h"

// One 256-lane tile reads one owned block and names its live elements by handle.
// Selection changes only which blocks and live bits participate.
inline uint MembershipElement(device const BindlessSet &b, ElementWorkSeedJob source, uint group, uint lane) {
    if (group >= source.BlockCount) return InvalidOffset;
    const uint id = BindlessBuffer(uint,b.Buffer,source.BlockIds.Slot)[source.BlockIds.Offset+group];
    const auto block = BindlessBuffer(MeshElementBlock,b.Buffer,source.BlocksSlot)[id];
    if (block.Owner != source.Owner) return InvalidOffset;
    uint live = block.Live[lane/32u];
    if (source.SelectionSlot != InvalidSlot) live &= BindlessBuffer(uint,b.Buffer,source.SelectionSlot)[id*8u+lane/32u];
    return live & (1u<<(lane%32u)) ? block.Local*256u+lane : InvalidOffset;
}

#endif
