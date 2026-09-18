#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Stores one mesh's arena locations once, shared by every primitive and instance that draws it.
// Corner offsets locate the mesh's first corner, and ComposeDraw advances them to a primitive's.
struct MeshRecord {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT();
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t ObjectIdSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassOffset DEFAULT(InvalidOffset);
    uint32_t CustomCornerMaskOffset DEFAULT(InvalidOffset);
    uint32_t CustomCornerNormalOffset DEFAULT(InvalidOffset);
    uint32_t BaseSeamNormalOffset DEFAULT(InvalidOffset);
    uint32_t CornerTangentOffset DEFAULT(InvalidOffset);
    uint32_t CornerColorOffset DEFAULT(InvalidOffset);
    GpuArray<uint32_t, 4> CornerUvOffsets DEFAULT(InvalidOffset, InvalidOffset, InvalidOffset, InvalidOffset);
    uint32_t FaceIdOffset DEFAULT();
    uint32_t BaseFaceNormalOffset DEFAULT(InvalidOffset);
    uint32_t FaceFirstTriangleOffset DEFAULT(InvalidOffset);
    uint32_t VertexEdgeAdjacencyOffset DEFAULT(InvalidOffset);
    uint32_t VertexFanAdjacencyOffset DEFAULT(InvalidOffset);
    SlotOffset Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t ConnectivityFaceStarts DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    uint32_t EditEdgeOffset DEFAULT(InvalidOffset);
    uint32_t InstanceStateSlot DEFAULT(InvalidSlot);
    uint32_t VertexOffset DEFAULT();
    uint32_t MorphShadingAuthored DEFAULT();
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    uint32_t ElementPrimitiveOffset DEFAULT(InvalidOffset);
};
static_assert(sizeof(MeshRecord) == 128, "MeshRecord size");
