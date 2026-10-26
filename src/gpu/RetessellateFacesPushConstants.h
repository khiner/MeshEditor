#pragma once
#include "gpu/ElementTables.h"
#include "gpu/ElementWork.h"

// Sparse face handles and their offsets into reusable polygon scratch, with the record's tables resolving its handles.
struct RetessellateFacesPushConstants {
    ElementTables Tables DEFAULT();
    SlotOffset Faces DEFAULT(), Scratch DEFAULT();
    ElementWork ChangedTriangles DEFAULT(), ChangedVertices DEFAULT();
    SlotOffset Tangents DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t VertexSlot DEFAULT(), CornerSlot DEFAULT(), FaceRangesSlot DEFAULT(), FaceTrianglesSlot DEFAULT(), TriangleSlot DEFAULT();
};
