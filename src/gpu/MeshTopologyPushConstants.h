#pragma once
#include "gpu/Types.h"
// Shared topology-operator slots with per-pass tiles beginning at FirstTile.
struct MeshTopologyPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
    uint32_t PassParameter DEFAULT(); // The scan a scan pass runs
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot);
    uint32_t ConnectivitySlot DEFAULT(InvalidSlot);
    uint32_t SelectionBitsSlot DEFAULT(InvalidSlot);
    uint32_t FaceFirstTriangleSlot DEFAULT(InvalidSlot);
    uint32_t TriangleFaceIdSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t FaceSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t ElementPrimitiveSlot DEFAULT(InvalidSlot);
    uint32_t BoneDeformSlot DEFAULT(InvalidSlot);
    uint32_t MorphTargetSlot DEFAULT(InvalidSlot);
    uint32_t CornerTangentSlot DEFAULT(InvalidSlot);
    uint32_t CornerColorSlot DEFAULT(InvalidSlot);
    uint32_t CornerUvSlot DEFAULT(InvalidSlot);
    uint32_t CustomCornerMaskSlot DEFAULT(InvalidSlot);
    uint32_t CustomCornerNormalSlot DEFAULT(InvalidSlot);
    uint32_t AdjacencySlot DEFAULT(InvalidSlot);
    uint32_t BaseVertexNormalSlot DEFAULT(InvalidSlot);
    uint32_t BaseFaceNormalSlot DEFAULT(InvalidSlot);
    uint32_t ListSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshTopologyPushConstants) == 100, "MeshTopologyPushConstants size");
