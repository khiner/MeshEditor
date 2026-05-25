#pragma once

#include "Armature.h"

// Registry-mutating armature edit operations: bone selection, structure rebuild, extrude/duplicate.

void SelectBone(entt::registry &, entt::entity); // Exclusive

// Reset selected bones' transforms to rest pose.
void ClearSelectedBoneTransforms(entt::registry &, bool position, bool rotation, bool scale);

// Finalize armature structure after AddBone/RemoveBone. Resets pose state and re-resolves animation indices.
void RebuildArmatureStructure(entt::registry &, entt::entity arm_data_entity);

struct ExtrudeResult {
    std::vector<BoneId> NewBoneIds;
    std::vector<uint32_t> UpdatedParentIndices;
};
ExtrudeResult ExtrudeBones(entt::registry &, Armature &, entt::entity arm_obj_entity);

struct DuplicateResult {
    struct Entry {
        entt::entity OriginalEntity;
        BoneId NewId;
    };
    std::vector<Entry> Duplicated;
};
DuplicateResult DuplicateBones(entt::registry &, Armature &, entt::entity arm_obj_entity);
