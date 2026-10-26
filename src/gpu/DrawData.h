#pragma once
#include "gpu/ConnectivityRef.h"

#include "gpu/EditSelectionStorage.h"

// Mesh and instance state.
// Stored references are record-relative handles, which the connectivity's tables resolve, and do not depend on primitive ranges.
struct DrawData {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT(); // Corner vertex handles, or an extras record's bone and joint index run
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t FirstInstance DEFAULT();
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassMode DEFAULT(InvalidOffset);
    // The mesh's layers the draw reads at its handles, each without a slot where the draw's topology has none.
    SlotOffset CustomNormals DEFAULT();
    SlotOffset CornerTangent DEFAULT();
    SlotOffset CornerColor DEFAULT();
    GpuArray<SlotOffset, 4> CornerUvs DEFAULT();
    // Base face normals mirror face-arena indexing.
    ConnectivityRef Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    EditSelectionStorage Selection DEFAULT();
    uint32_t PrimaryEditInstanceIndex DEFAULT(InvalidOffset);
    uint32_t BoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t ArmatureDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphWeightsOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    // Authored morph normals combine base normals with weighted authored deltas.
    // Edit-mode draws derive normals because edit mode creates no deform slots.
    uint32_t MorphShadingAuthored DEFAULT();
    // Current-pose values keyed by vertex, sector root and face arena addresses.
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    uint32_t MorphNormalNamespace DEFAULT(InvalidOffset);
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    SlotOffset ElementPrimitives DEFAULT();
    SlotOffset Skin DEFAULT(), Morph DEFAULT();
    SlotOffset CornerSectors DEFAULT(), NormalSectors DEFAULT();
};
