#pragma once

#include "gpu/Types.h"

// One instance's own draw state.
// Its mesh's record holds everything its instances share.
struct InstanceRecord {
    uint32_t Mesh DEFAULT(InvalidOffset);
    uint32_t ObjectId DEFAULT();
    uint32_t ArmatureDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphWeightsOffset DEFAULT(InvalidOffset);
    // Pose namespaces of an instance deformed apart from its mesh's other instances.
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    uint32_t MeshletBoundsNamespace DEFAULT(InvalidOffset);
    uint32_t MorphNormalNamespace DEFAULT(InvalidOffset);
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    uint32_t ExcitedVertex DEFAULT(InvalidOffset);
};
static_assert(sizeof(InstanceRecord) == 44, "InstanceRecord size");
