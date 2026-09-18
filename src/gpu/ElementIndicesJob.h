#pragma once
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"
// Writes one mesh's draw indices from its connectivity run: two vertex indices per edge and three per fan triangle.
// A stream the mesh does not need has a zero count.
struct ElementIndicesJob {
    SlotOffset Corners DEFAULT();
    SlotOffset Connectivity DEFAULT();
    SlotOffset Endpoints DEFAULT();
    SlotOffset Triangles DEFAULT();
    SlotOffset TriangleFaceIds DEFAULT();
    SlotOffset FaceFirstTriangles DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t EdgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t TriangleCount DEFAULT();
    uint32_t FaceStarts DEFAULT();
};
static_assert(sizeof(ElementIndicesJob) == 72, "ElementIndicesJob size");
