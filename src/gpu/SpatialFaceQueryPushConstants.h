#pragma once

#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Canonical candidate face membership and an exact geometric predicate.
struct SpatialFaceQueryPushConstants {
    ElementWork Candidates DEFAULT(), Faces DEFAULT();
    uint32_t CandidateCount DEFAULT();
    uint32_t FaceRangeSlot DEFAULT(InvalidSlot), CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
    uint32_t FaceRangeCapacity DEFAULT(), CornerCapacity DEFAULT(), VertexCapacity DEFAULT();
    uint32_t ResultSlot DEFAULT(InvalidSlot), ResultOffset DEFAULT();
    uint32_t Mode DEFAULT(); // 0 cuts by the plane, 1 deletes by plane side, and 2 selects by screen stroke.
    vec3 PlaneNormal DEFAULT();
    float PlaneOffset DEFAULT();
    mat4 ScreenTransform DEFAULT();
    vec2 Extent DEFAULT(), KnifeStart DEFAULT(), KnifeEnd DEFAULT();
};
