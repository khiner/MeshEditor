#pragma once

#include "gpu/EditSelectionStorage.h"
#include "gpu/Types.h"

struct InstanceRecord {
    uint32_t PrimitiveOffset DEFAULT(InvalidOffset);
    uint32_t PrimitiveCount DEFAULT();
    uint32_t BoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t ArmatureDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphWeightsOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    uint32_t PosedPositionOffset DEFAULT(InvalidOffset);
    uint32_t PosedVertexNormalOffset DEFAULT(InvalidOffset);
    uint32_t PosedSeamNormalOffset DEFAULT(InvalidOffset);
    uint32_t PosedFaceNormalOffset DEFAULT(InvalidOffset);
    uint32_t PosedMeshletBoundsOffset DEFAULT(InvalidOffset);
    EditSelectionStorage Selection DEFAULT();
    uint32_t EditEdgeSharpnessOffset DEFAULT(InvalidOffset);
    uint32_t HasPendingVertexTransform DEFAULT();
    uint32_t PrimaryEditInstanceIndex DEFAULT(InvalidOffset);
    uint32_t ObjectId DEFAULT();
    uint32_t Flags DEFAULT();
    uint32_t ElementIdOffset DEFAULT();
    uint32_t ActiveVertex DEFAULT(InvalidOffset);
    uint32_t ExcitedVertex DEFAULT(InvalidOffset);
};
static_assert(sizeof(InstanceRecord) == 112, "InstanceRecord size");
