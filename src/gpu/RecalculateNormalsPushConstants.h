#pragma once
#include "gpu/ConnectivityRef.h"

struct RecalculateNormalsPushConstants {
    ConnectivityRef Connectivity;
    SlotOffset Faces, Groups, Tiles, PartialCenters, Centers, Candidates, Orientations;
    uint32_t VertexSlot, CornerSlot, NormalSlot, FaceCount;
};

struct NormalOrientationCandidate {
    vec3 Score;
    uint32_t Rank, Flip;
};
