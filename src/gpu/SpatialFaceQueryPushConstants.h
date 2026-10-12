#pragma once

#include "gpu/ElementTables.h"
#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Candidate face membership and an exact geometric predicate.
// The record's tables resolve its handles, each below its domain's bound.
struct SpatialFaceQueryPushConstants {
    ElementWork Candidates DEFAULT(), Faces DEFAULT();
    ElementTables Tables DEFAULT();
    uint32_t CandidateCount DEFAULT();
    uint32_t FaceRangeSlot DEFAULT(InvalidSlot), CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
    uint32_t FaceBound DEFAULT(), CornerBound DEFAULT(), VertexBound DEFAULT();
    uint32_t ResultSlot DEFAULT(InvalidSlot), ResultOffset DEFAULT();
    uint32_t Mode DEFAULT(); // 0 cuts by the plane, 1 deletes by plane side, and 2 selects by screen stroke.
    vec3 PlaneNormal DEFAULT();
    float PlaneOffset DEFAULT();
    mat4 ScreenTransform DEFAULT();
    vec2 Extent DEFAULT(), KnifeStart DEFAULT(), KnifeEnd DEFAULT();
};
