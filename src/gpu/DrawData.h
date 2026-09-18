#pragma once

#include "gpu/EditSelectionStorage.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// The draw context a shader works in: a mesh record advanced to one primitive, with one instance's state.
struct DrawData {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT();
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t FirstInstance DEFAULT();
    uint32_t ObjectIdSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassOffset DEFAULT(InvalidOffset);
    // Sparse corner normals use one presence/rank pair per 32 corners and one packed offset per present corner.
    // CornerBase locates the draw's first corner in the masks.
    uint32_t CustomCornerMaskOffset DEFAULT(InvalidOffset);
    uint32_t CustomCornerNormalOffset DEFAULT(InvalidOffset);
    uint32_t CornerBase DEFAULT();
    // Sector normals use the low bits of a Seam class value as the index.
    uint32_t BaseSeamNormalOffset DEFAULT(InvalidOffset);
    uint32_t CornerTangentOffset DEFAULT(InvalidOffset);
    uint32_t CornerColorOffset DEFAULT(InvalidOffset);
    GpuArray<uint32_t, 4> CornerUvOffsets DEFAULT(InvalidOffset, InvalidOffset, InvalidOffset, InvalidOffset);
    uint32_t FaceIdOffset DEFAULT();
    // Base face normals mirror face-arena indexing.
    uint32_t BaseFaceNormalOffset DEFAULT(InvalidOffset);
    uint32_t FaceFirstTriangleOffset DEFAULT(InvalidOffset);
    uint32_t VertexEdgeAdjacencyOffset DEFAULT(InvalidOffset);
    uint32_t VertexFanAdjacencyOffset DEFAULT(InvalidOffset);
    SlotOffset Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t ConnectivityFaceStarts DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    uint32_t ElementIdOffset DEFAULT();
    EditSelectionStorage Selection DEFAULT();
    uint32_t EditEdgeOffset DEFAULT(InvalidOffset);
    uint32_t InstanceStateSlot DEFAULT(InvalidSlot);
    uint32_t HasPendingVertexTransform DEFAULT();
    uint32_t PrimaryEditInstanceIndex DEFAULT(InvalidOffset);
    uint32_t VertexOffset DEFAULT();
    uint32_t BoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t ArmatureDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphWeightsOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    // Authored morph normals combine base normals with weighted authored deltas.
    // Edit-mode draws derive normals because edit mode creates no deform slots.
    uint32_t MorphShadingAuthored DEFAULT();
    // Current-pose mesh-local positions and their derived vertex and sector normals.
    uint32_t PosedPositionOffset DEFAULT(InvalidOffset);
    uint32_t PosedVertexNormalOffset DEFAULT(InvalidOffset);
    uint32_t PosedSeamNormalOffset DEFAULT(InvalidOffset);
    uint32_t PosedFaceNormalOffset DEFAULT(InvalidOffset);
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    uint32_t ElementPrimitiveOffset DEFAULT(InvalidOffset);
};
static_assert(sizeof(DrawData) == 216, "DrawData size");
