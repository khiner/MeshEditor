#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementAttributeRef.h"

#include "gpu/EditSelectionStorage.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Mesh and instance state. Canonical references do not depend on primitive ranges.
struct DrawData {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT();
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t FirstInstance DEFAULT();
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassMode DEFAULT(InvalidOffset);
    ElementAttributeRef CustomNormals DEFAULT();
    ElementAttributeRef CornerTangent DEFAULT();
    ElementAttributeRef CornerColor DEFAULT();
    GpuArray<ElementAttributeRef, 4> CornerUvs DEFAULT();
    // Base face normals mirror face-arena indexing.
    ConnectivityRef Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    uint32_t ElementIdOffset DEFAULT();
    EditSelectionStorage Selection DEFAULT();
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
    // Current-pose values keyed by canonical vertex, sector record, and face.
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    uint32_t MorphNormalNamespace DEFAULT(InvalidOffset);
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    ElementAttributeRef ElementPrimitives DEFAULT();
};
static_assert(sizeof(DrawData) == 264, "DrawData size");
