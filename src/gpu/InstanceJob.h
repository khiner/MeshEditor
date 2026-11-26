#pragma once
#include "gpu/NamedAttributeRef.h"
#include "gpu/SlotOffset.h"

// Writes each live point of one points record's transform, at its rank among the points, from its position, rotation and scale, one tile per block.
// Blocks holds the points' blocks in local order followed by their inclusive live prefixes.
struct InstanceTransformsJob {
    SlotOffset Blocks DEFAULT(), Transforms DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t VertexSlot DEFAULT(), MembershipSlot DEFAULT();
    NamedAttributeRef Rotations DEFAULT(), Scales DEFAULT();
};

// Moves the live vertices of Count consecutive arena addresses from First, where address First + i belongs to copy i / Bound, by that copy's transform, one thread per address.
struct PlaceCopiesJob {
    SlotOffset Transforms DEFAULT();
    uint32_t First DEFAULT(), Bound DEFAULT(), Count DEFAULT();
    uint32_t VertexSlot DEFAULT(), MembershipSlot DEFAULT();
};
