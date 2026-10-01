#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshletIndex.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// The meshlet spatial tree supplies conservative candidates. The final
// face work contains only faces whose canonical edge or centroid predicate
// matches the requested topology operation.
struct SpatialFaceQueryPushConstants {
    MeshletIndexRef Meshlets DEFAULT();
    ElementWork Faces DEFAULT();
    uint32_t CandidateCount DEFAULT();
    SlotOffset MeshletCandidates DEFAULT();
    uint32_t MeshletCandidateCount DEFAULT();
    uint32_t SpatialRoot DEFAULT(InvalidOffset), SpatialNodeSlot DEFAULT(InvalidSlot);
    uint32_t SpatialNodeCapacity DEFAULT(), SeedDepth DEFAULT();
    uint32_t MeshletSlot DEFAULT(InvalidSlot), TriangleIdSlot DEFAULT(InvalidSlot);
    uint32_t TriangleSlot DEFAULT(InvalidSlot), HalfedgeFaceSlot DEFAULT(InvalidSlot);
    uint32_t FaceRangeSlot DEFAULT(InvalidSlot), CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
    uint32_t FaceBlockSlot DEFAULT(InvalidSlot), FaceOwner DEFAULT();
    uint32_t FaceCapacity DEFAULT(), FaceRangeCapacity DEFAULT(), TriangleCapacity DEFAULT();
    uint32_t CornerCapacity DEFAULT(), VertexCapacity DEFAULT();
    uint32_t TriangleIdCapacity DEFAULT(), MeshletCapacity DEFAULT();
    uint32_t ResultSlot DEFAULT(InvalidSlot), ResultOffset DEFAULT();
    uint32_t Mode DEFAULT(); // 0 cuts by the plane, 1 deletes by plane side, and 2 selects by screen stroke.
    vec3 PlaneNormal DEFAULT();
    float PlaneOffset DEFAULT();
    mat4 ScreenTransform DEFAULT();
    vec2 Extent DEFAULT(), KnifeStart DEFAULT(), KnifeEnd DEFAULT();
};
