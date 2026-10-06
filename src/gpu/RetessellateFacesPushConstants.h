#pragma once
#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Transform.h"

// Sparse face handles and their offsets into reusable polygon scratch.
struct RetessellateFacesPushConstants {
    SlotOffset Faces DEFAULT(), Scratch DEFAULT();
    ElementWork ChangedTriangles DEFAULT(), ChangedVertices DEFAULT();
    ElementAttributeRef Tangents DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t VertexSlot DEFAULT(), CornerSlot DEFAULT(), FaceRangesSlot DEFAULT(), FaceTrianglesSlot DEFAULT(), TriangleSlot DEFAULT();
    uint32_t SelectionSlot DEFAULT(InvalidSlot), ApplyTransform DEFAULT();
    Transform Primary DEFAULT(), Delta DEFAULT();
    vec3 Pivot DEFAULT();
};
static_assert(sizeof(RetessellateFacesPushConstants) == 180);
