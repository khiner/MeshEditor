#pragma once
#include "gpu/MeshAttributeBit.h"
#include "gpu/ElementWork.h"
#include "gpu/ConnectivityRef.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"
// One topology operator run: the source mesh, the output ranges, and the scratch layout.
// The output fields are filled after the count scan, once the host has allocated the output at its exact counts.
// Arena offsets index the arenas the push constants name by slot.
struct MeshTopologyJob {
    MeshTopologyOp Op DEFAULT();
    uint32_t Flags DEFAULT();
    uint32_t Steps DEFAULT(1); // Extrude region: layers of copies, each moved through the transform once more
    float Param0 DEFAULT();
    float Param1 DEFAULT();
    // The source vertex a merge keeps and its output position.
    uint32_t TargetVertex DEFAULT(InvalidOffset);
    vec3 TargetPosition DEFAULT();
    // The transform applied to copies, as three columns and a translation.
    mat3 CopyRotation DEFAULT();
    vec3 CopyTranslation DEFAULT();
    // A plane as its unit normal and offset, positive on the normal's side.
    vec3 PlaneNormal DEFAULT();
    float PlaneOffset DEFAULT();
    // A knife: the mesh-to-clip transform, the target extent in pixels, and the segment in those pixels.
    mat4 ScreenTransform DEFAULT();
    vec2 Extent DEFAULT();
    vec2 KnifeStart DEFAULT();
    vec2 KnifeEnd DEFAULT();
    // Source mesh.
    ConnectivityRef SrcConnectivity DEFAULT();
    uint32_t SrcVertexCount DEFAULT();
    uint32_t SrcHalfedgeCount DEFAULT();
    uint32_t SrcFaceCount DEFAULT();
    uint32_t SrcEdgeCount DEFAULT();
    // Compact work indices resolve to canonical arena handles without staging geometry.
    ElementWork SrcVertexWork DEFAULT(), SrcHalfedgeWork DEFAULT(), SrcFaceWork DEFAULT(), SrcEdgeWork DEFAULT();
    ElementWork PrimitiveWork DEFAULT(); // Fresh output remaps source primitive IDs to its compact palette.
    SlotOffset SrcVertexBits DEFAULT();
    SlotOffset SrcEdgeBits DEFAULT();
    SlotOffset SrcFaceBits DEFAULT();
    uint32_t HasSkin DEFAULT();
    uint32_t CornerAttributes DEFAULT();
    uint32_t VertexAttributes DEFAULT();
    // A face list: a count, then each face's length and vertex indices.
    uint32_t ListOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    uint32_t CollapseCount DEFAULT();
    SlotOffset CollapseVertices DEFAULT(); // The selected vertex handles a collapse ranks, or none when it collapses every source vertex
    // Output mesh.
    uint32_t DstCornerOffset DEFAULT();
    ConnectivityRef DstConnectivity DEFAULT();
    uint32_t DstVertexCount DEFAULT();
    uint32_t DstHalfedgeCount DEFAULT();
    uint32_t DstFaceCount DEFAULT();
    // Compact-index -> canonical-handle maps. Corners and triangles
    // are emitted into independently reserved contiguous runs.
    SlotOffset DstVertexHandles DEFAULT(), DstFaceHandles DEFAULT();
    SlotOffset DstVertexBits DEFAULT();
    SlotOffset DstEdgeBits DEFAULT();
    SlotOffset DstFaceBits DEFAULT();
    uint32_t DstTriangleOffset DEFAULT();
    uint32_t DstEdgeSharpnessOffset DEFAULT();
    // Scratch layout, in words.
    uint32_t StateOffset DEFAULT(); // Two words: whether an extruded region borders unselected faces, then whether a label pass changed a value this round
    uint32_t FlagVertexOffset DEFAULT(); // Per source vertex: tagged, kept, region, and copy bits
    uint32_t VertexTargetOffset DEFAULT(); // Per source vertex: the source vertex its corners map to
    uint32_t FlagHalfedgeOffset DEFAULT(); // Per source halfedge
    uint32_t FlagFaceOffset DEFAULT(); // Per source face: 1 when the face survives
    // Three count arrays over N = source vertices + halfedges + faces + 1, scanned in place: vertices, faces, corners.
    uint32_t CountsOffset DEFAULT();
    uint32_t CountEntries DEFAULT();
    uint32_t CountBlockOffset DEFAULT();
    uint32_t CountBlockCount DEFAULT();
    // Per output vertex: source a, source b, and the weight of b as float bits.
    uint32_t VertexMapOffset DEFAULT();
    // Per output halfedge: source corners a and b, the weight of b as float bits, the source halfedge whose edge the output edge inherits, and 1 when the edge is selected.
    uint32_t CornerMapOffset DEFAULT();
    uint32_t FaceMapOffset DEFAULT(); // Per output face: source face or InvalidOffset
    uint32_t CornerProvenanceOffset DEFAULT(); // Custom normals only: compact output vertex and face per corner
    // Iterating operators: per source face, its label, its region's boundary halfedge count and lowest boundary halfedge, and its walked loop length.
    // Then per source vertex, its edge count and its dissolved edge count.
    uint32_t LabelOffset DEFAULT();
    // Per source halfedge: an edge split's sector representative, or per source edge, a listed cut's parameter as float bits or InvalidOffset.
    uint32_t HalfedgeAuxOffset DEFAULT();
    // Per affected face: expanded subdivision loop, chords, and face-walk work.
    uint32_t FaceLoopOffset DEFAULT();
    // An open-addressing table with its mask: a merge by distance's vertex indices keyed by grid cell, then a joining line core's lowest representative corner per output line.
    uint32_t TableOffset DEFAULT();
    uint32_t TableMask DEFAULT();
    // Collapse: selected keys/order, radix histograms, and tiled segmented sums.
    uint32_t CollapseOffset DEFAULT();
    // Per output vertex: two inward vectors, or one displacement and a zero, that the gather adds to its position.
    uint32_t VertexInwardOffset DEFAULT();
    // Retained neighborhood corners whose authored normals need rebasing after
    // their derived frames change. SrcHalfedgeWork excludes replaced loops.
    ElementWork RetainedNormalCorners DEFAULT();
    uint32_t RetainedNormalCornerCount DEFAULT();
    // Optional emitted-triangle ordinal -> original triangle, for local render ownership.
    SlotOffset DstTriangleSources DEFAULT();
    // Optional output-vertex basis for staged inset parameter updates.
    SlotOffset DstInsetBasis DEFAULT();
};
static_assert(sizeof(MeshTopologyJob) == 640, "MeshTopologyJob size");
