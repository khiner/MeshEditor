#pragma once
#include "gpu/ConnectivityRef.h"

// A point counts as outside a face beyond this fraction of its set's extent, so coplanar points stay inside and every cone face has a stable normal.
GPU_CONSTANT float HullTolerance = 1e-4f;
// A set whose farthest point from its seed plane is within this, in the plane normal's scale, is flat.
GPU_CONSTANT float HullFlat = 1e-12f;

// A point's face word when it awaits its first face, lies inside the hull, or is a hull vertex.
GPU_CONSTANT uint32_t HullPointUnassigned = ~0u;
GPU_CONSTANT uint32_t HullPointInside = ~0u - 1u;
GPU_CONSTANT uint32_t HullPointUsed = ~0u - 2u;

// A set is open while its hull grows, flat when its points lie on a line or coincide, and planar when its points lie in a plane.
// A set is abandoned when a horizon is not one simple loop.
// A planar set's hull is the pyramid from an apex lifted along the plane normal, whose lateral faces hold the plane hull's edges.
GPU_CONSTANT uint32_t HullSetOpen = 0u;
GPU_CONSTANT uint32_t HullSetFlat = 1u;
GPU_CONSTANT uint32_t HullSetAbandoned = 2u;
GPU_CONSTANT uint32_t HullSetPlanar = 3u;

struct HullSet {
    uint32_t First DEFAULT(); // First point in the job's positions, and the next set's First is its end
    uint32_t Status DEFAULT();
    // Per seed stage: the ordered bits of the farthest distance from the first point, then the line through two seeds, then the plane through three.
    GpuArray<uint32_t, 3> Best DEFAULT();
    GpuArray<uint32_t, 3> Seed DEFAULT(); // The point farthest in each stage
    float Tolerance DEFAULT(); // HullTolerance times the extent
};

// A hull face, whose plane is its unnormalized normal and offset, and whose tolerance is the normal's length times its set's.
struct HullFace {
    uvec3 V DEFAULT(); // Point indices, wound outward
    uvec3 Across DEFAULT(); // The face across each edge V[k] to V[k + 1]
    vec4 Plane DEFAULT();
    float Tolerance DEFAULT();
    uint32_t Set DEFAULT();
    // The ordered height bits of the farthest point outside the face and that point, which the face proposes as an apex.
    uint32_t ApexHeight DEFAULT(), Apex DEFAULT(InvalidOffset);
    // The proposing face whose apex's visible region holds this face, with the lowest face winning, and the region flood's worklist link.
    uint32_t Claim DEFAULT(InvalidOffset), Next DEFAULT(InvalidOffset);
    uint32_t Accepted DEFAULT();
    uint32_t FirstNew DEFAULT(InvalidOffset); // The lowest new face of this face's apex's cone
};

// Scratch word offsets of a quickhull over point sets, which the host drives in rounds.
struct HullJob {
    uint32_t Sets DEFAULT(), Points DEFAULT();
    uint32_t SetRecords DEFAULT(); // Per set: HullSet
    uint32_t Positions DEFAULT(); // Per point then per set: packed_float3, the points then each set's lifted apex
    uint32_t PointFace DEFAULT(); // Per point: the face it lies outside of, or its unassigned, inside or used marker
    uint32_t Proposals DEFAULT(); // One word: the faces proposing an apex this round
    uint32_t Faces DEFAULT(), Survivors DEFAULT(), NewFaces DEFAULT();
    uint32_t Current DEFAULT(), Next DEFAULT(); // Per face: HullFace
    // Per current face then a terminator: the horizon edges it emits, and whether it survives, each scanned in place.
    uint32_t NewCount DEFAULT(), Survivor DEFAULT();
};

// A point set of pairwise sums: its first point, and its vertex handles in each handle list.
struct HullSumSet {
    uint32_t First DEFAULT();
    uint32_t FirstA DEFAULT(), FirstB DEFAULT(), CountB DEFAULT();
};

// The positions a hull gathers from one record's listed vertices, or sums from two records' listed vertices set by set.
struct HullPointsPushConstants {
    SlotOffset Positions DEFAULT(); // Per point: packed_float3
    uint32_t Count DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    ConnectivityRef First DEFAULT(), Second DEFAULT();
    SlotOffset FirstHandles DEFAULT(), SecondHandles DEFAULT();
    SlotOffset Sets DEFAULT(); // Per set: HullSumSet
    uint32_t SetCount DEFAULT();
    float SecondSign DEFAULT(1.f); // The sign the second record's positions enter the sums with, negative to sweep by the reflected record
    vec3 SecondOffset DEFAULT(); // Added to every sum, which recenters a reflected brush about a point of its own
};
