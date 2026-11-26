#pragma once
#include "Bindless.metal"
#include "gpu/ElementWorkSeedJob.h"

// One 256-lane tile reads one owned block and returns each live element's handle in the job's range, or InvalidOffset.
inline uint MembershipElement(device const BindlessSet &b, ElementWorkSeedJob source, uint group, uint lane) {
    if (group >= source.BlockCount) return InvalidOffset;
    const uint id = BindlessBuffer(uint,b.Buffer,source.BlockIds.Slot)[source.BlockIds.Offset+group];
    const auto block = BindlessBuffer(MeshElementBlock,b.Buffer,source.BlocksSlot)[id];
    if (block.Owner != source.Owner) return InvalidOffset;
    uint live = block.Live[lane/32u];
    if (source.SelectionSlot != InvalidSlot) live &= BindlessBuffer(uint,b.Buffer,source.SelectionSlot)[id*MeshElementBlockWords+lane/32u];
    const uint handle=block.Local*MeshElementBlockSize+lane;
    return live & (1u<<(lane%32u)) && handle>=source.First && handle<source.End ? handle : InvalidOffset;
}
