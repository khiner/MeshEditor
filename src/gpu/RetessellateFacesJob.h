#pragma once
#include "gpu/ElementTables.h"
#include "gpu/ElementWork.h"

// Sparse face handles and their offsets into reusable polygon scratch, with the record's tables resolving its handles.
// A job without face handles covers every face of a dense record, writes its first-triangle table and the triangles of its faces of up to PolygonThreadCorners corners, and leaves larger polygons to a job with handles.
// A job runs one thread per face.
struct RetessellateFacesJob {
    ElementTables Tables DEFAULT();
    SlotOffset Faces DEFAULT(), Scratch DEFAULT();
    ElementWork ChangedTriangles DEFAULT(), ChangedVertices DEFAULT();
    SlotOffset Tangents DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t VertexSlot DEFAULT(), CornerSlot DEFAULT(), FaceRangesSlot DEFAULT(), FaceTrianglesSlot DEFAULT(), TriangleSlot DEFAULT();
};
