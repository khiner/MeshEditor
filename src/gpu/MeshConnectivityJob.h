#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"
// Defines a connectivity rebuild's canonical work and scratch layout.
// Work ordinals index scratch.
// Every persistent reference is a canonical handle.
// The caller reserves enough edge handles, then publishes the resulting count.
struct MeshConnectivityJob {
    SlotOffset Corners DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    // Sparse work contains canonical handles. An implicit range begins at the
    // corresponding connectivity offset (or Corners.Offset for halfedges).
    ElementWork Vertices DEFAULT(), Halfedges DEFAULT(), Faces DEFAULT();
    ElementHandleRange EdgeHandles DEFAULT(); // New-edge ordinal -> canonical handle, from a run or a list.
    // The old affected edges read source clones when canonical data is
    // overwritten. Matching face endpoint pairs or explicit wire pairs keep their canonical handles.
    ConnectivityRef SourceConnectivity DEFAULT();
    uint32_t SourceCornerSlot DEFAULT(InvalidSlot);
    ElementWork SourceEdges DEFAULT();
    ElementWork ConvertedEdges DEFAULT(); // Explicit changes between surface and loose-edge ownership.
    SlotOffset ConvertedWirePairs DEFAULT(); // Per converted-edge ordinal: wire pair's first halfedge + 1, zero for a surface, or InvalidOffset to retire. Absent means all become surfaces.
    uint32_t SourceEdgeCount DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    // Nonzero when the run stores each face's first halfedge, which a mesh whose faces are not all triangles needs.
    uint32_t FaceStarts DEFAULT();
    uint32_t WordCount DEFAULT();
    uint32_t SourceWordCount DEFAULT();
    uint32_t ScanWordCount DEFAULT(); // New-edge words + terminator + retired-edge words + terminator.
    // Open-addressed table keyed by endpoint pair, each slot holding the lowest halfedge of its undirected edge.
    uint32_t TableOffset DEFAULT();
    uint32_t TableMask DEFAULT();
    // Each halfedge's table slot during insertion, then the lowest halfedge of its edge.
    uint32_t RepOffset DEFAULT();
    // Per representative, the lowest halfedge running against it on its edge.
    uint32_t PartnerOffset DEFAULT();
    uint32_t PopcountOffset DEFAULT();
    uint32_t WordBlockOffset DEFAULT();
    uint32_t WordBlockCount DEFAULT();
    // Bits contain new representatives then retired source edges. Ranks have
    // an additional zero-count terminator after each of those two domains.
    uint32_t BitsOffset DEFAULT();
    uint32_t RanksOffset DEFAULT();
    uint32_t RetainedEdgesOffset DEFAULT(InvalidOffset); // Per compact destination representative, an old edge or InvalidOffset.
    uint32_t RetiredEdgesOffset DEFAULT(); // Dense canonical retirement handles, in source-work order.
    // Two counts: newly required edges and retired source edges.
    uint32_t StateOffset DEFAULT();
    ElementWork RetiredEdgeWork DEFAULT(); // Optional canonical retirement set, emitted with the dense scratch list.
};
static_assert(sizeof(MeshConnectivityJob) == 332, "MeshConnectivityJob size");
