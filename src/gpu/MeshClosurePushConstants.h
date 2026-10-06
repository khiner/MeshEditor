#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Closure passes over one mesh's canonical connectivity, reading finished input work by ordinal below InputBound.
// Vertex input (domain 0) marks its fans, face input (domain 2) marks its loops, and edge input (domain 3) marks its endpoints.
struct MeshClosurePushConstants {
    ConnectivityRef Connectivity DEFAULT();
    GpuArray<ElementWork, 4> Work DEFAULT(); // Vertices, halfedges, faces, edges.
    ElementWork Input DEFAULT(), Retained DEFAULT();
    uint32_t CornerSlot DEFAULT(InvalidSlot), FaceCount DEFAULT();
    uint32_t InputDomain DEFAULT(), InputBound DEFAULT(), RetainedBound DEFAULT();
    uint32_t RetainIsolatedOnly DEFAULT(); // Filter retained vertex seeds to vertices without incident edges.
    SlotOffset Incidence DEFAULT(); // The word counting the input's fan corners, loop corners, or endpoint fan corners
};
static_assert(sizeof(MeshClosurePushConstants) == 188, "MeshClosurePushConstants size");

// The derived triangles of finished faces, checked against the mesh's face and triangle ownership.
struct FaceTrianglePushConstants {
    ElementWork Faces DEFAULT(), Triangles DEFAULT();
    SlotOffset Error DEFAULT(), Total DEFAULT();
    uint32_t FaceBound DEFAULT(), FaceOwner DEFAULT(), TriangleOwner DEFAULT();
    uint32_t FaceBlocksSlot DEFAULT(), TriangleBlocksSlot DEFAULT();
    uint32_t FaceRangesSlot DEFAULT(), FaceTrianglesSlot DEFAULT(), TrianglesSlot DEFAULT();
    uint32_t FaceCapacity DEFAULT(), TriangleCapacity DEFAULT(), CornerCapacity DEFAULT();
};
static_assert(sizeof(FaceTrianglePushConstants) == 92, "FaceTrianglePushConstants size");
