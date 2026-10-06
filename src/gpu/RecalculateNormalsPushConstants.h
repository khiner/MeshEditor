#pragma once
#include "gpu/ConnectivityRef.h"

struct RecalculateNormalsPushConstants {
    ConnectivityRef Connectivity;
    SlotOffset Faces, Groups, Tiles, PartialCenters, Centers, Candidates, Orientations;
    uint32_t VertexSlot, CornerSlot, NormalSlot, FaceCount;
};
static_assert(sizeof(RecalculateNormalsPushConstants) == 132);

struct NormalOrientationCandidate {
    vec3 Score;
    uint32_t Rank, Flip;
};
static_assert(sizeof(NormalOrientationCandidate) == 20);
