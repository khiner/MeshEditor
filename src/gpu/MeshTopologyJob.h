#pragma once
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
    uint32_t SrcVertexOffset DEFAULT();
    uint32_t SrcCornerOffset DEFAULT();
    uint32_t SrcConnectivityOffset DEFAULT();
    uint32_t SrcVertexCount DEFAULT();
    uint32_t SrcHalfedgeCount DEFAULT();
    uint32_t SrcFaceCount DEFAULT();
    uint32_t SrcEdgeCount DEFAULT();
    uint32_t SrcFaceStarts DEFAULT();
    uint32_t SrcVertexBitsOffset DEFAULT();
    uint32_t SrcEdgeBitsOffset DEFAULT();
    uint32_t SrcFaceBitsOffset DEFAULT();
    uint32_t SrcFaceFirstTriangleOffset DEFAULT();
    uint32_t SrcEdgeSharpnessOffset DEFAULT();
    uint32_t SrcElementPrimitiveOffset DEFAULT();
    uint32_t SrcBoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t SrcMorphTargetOffset DEFAULT(InvalidOffset);
    uint32_t SrcCornerTangentOffset DEFAULT(InvalidOffset);
    uint32_t SrcCornerColorOffset DEFAULT(InvalidOffset);
    GpuArray<uint32_t, 4> SrcCornerUvOffsets DEFAULT(InvalidOffset, InvalidOffset, InvalidOffset, InvalidOffset);
    uint32_t SrcCustomCornerMaskOffset DEFAULT(InvalidOffset);
    uint32_t SrcCustomCornerNormalOffset DEFAULT(InvalidOffset);
    uint32_t SrcFanAdjacencyOffset DEFAULT(InvalidOffset);
    // Per source halfedge, its face, staged in scratch for a mesh whose faces are not all triangles.
    uint32_t SrcFaceOffset DEFAULT(InvalidOffset);
    // A face list: a count, then each face's length and vertex indices.
    uint32_t ListOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    // Output mesh.
    uint32_t DstVertexOffset DEFAULT();
    uint32_t DstCornerOffset DEFAULT();
    uint32_t DstConnectivityOffset DEFAULT();
    uint32_t DstVertexCount DEFAULT();
    uint32_t DstHalfedgeCount DEFAULT();
    uint32_t DstFaceCount DEFAULT();
    uint32_t DstFaceStarts DEFAULT();
    uint32_t DstVertexBitsOffset DEFAULT();
    uint32_t DstEdgeBitsOffset DEFAULT();
    uint32_t DstFaceBitsOffset DEFAULT();
    uint32_t DstFaceFirstTriangleOffset DEFAULT();
    uint32_t DstTriangleFaceIdOffset DEFAULT();
    uint32_t DstEdgeSharpnessOffset DEFAULT();
    uint32_t DstElementPrimitiveOffset DEFAULT();
    uint32_t DstBoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t DstMorphTargetOffset DEFAULT(InvalidOffset);
    uint32_t DstCornerTangentOffset DEFAULT(InvalidOffset);
    uint32_t DstCornerColorOffset DEFAULT(InvalidOffset);
    GpuArray<uint32_t, 4> DstCornerUvOffsets DEFAULT(InvalidOffset, InvalidOffset, InvalidOffset, InvalidOffset);
    uint32_t DstCustomCornerMaskOffset DEFAULT(InvalidOffset);
    uint32_t DstCustomCornerNormalOffset DEFAULT(InvalidOffset);
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
    // Iterating operators: per source face, its label, its region's boundary halfedge count and lowest boundary halfedge, and its walked loop length.
    // Then per source vertex, its edge count and its dissolved edge count.
    uint32_t LabelOffset DEFAULT();
    // Per source halfedge: an edge split's sector representative, or per source edge, a listed cut's parameter as float bits or InvalidOffset.
    uint32_t HalfedgeAuxOffset DEFAULT();
    // Merge by distance: an open-addressing table of vertex indices keyed by grid cell, with its mask.
    uint32_t TableOffset DEFAULT();
    uint32_t TableMask DEFAULT();
    // Collapse: per source vertex, a flag word and the center position of its run.
    uint32_t VertexOverrideOffset DEFAULT();
    // Per output vertex: two inward vectors, or one displacement and a zero, that the gather adds to its position.
    uint32_t VertexInwardOffset DEFAULT();
    // Per output fan-corner word plus one terminator: custom normal popcounts, scanned in place.
    uint32_t CustomPopcountOffset DEFAULT();
    uint32_t CustomWordCount DEFAULT();
    uint32_t CustomBlockOffset DEFAULT();
    uint32_t CustomBlockCount DEFAULT();
};
static_assert(sizeof(MeshTopologyJob) == 484, "MeshTopologyJob size");
