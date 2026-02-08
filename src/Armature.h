#pragma once

#include "Transform.h"
#include "entt_fwd.h"
#include "numeric/mat4.h"

#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using BoneId = uint32_t;

inline constexpr BoneId InvalidBoneId{0};
inline constexpr uint32_t InvalidBoneIndex{std::numeric_limits<uint32_t>::max()};

struct ArmatureBone {
    BoneId Id{InvalidBoneId}; // Stable identifier - never reused
    uint32_t ParentIndex{InvalidBoneIndex}, FirstChild{InvalidBoneIndex}, NextSibling{InvalidBoneIndex}; // Dense runtime indices

    std::string Name;
    Transform RestLocal{};
    mat4 RestWorld{I4}, InvRestWorld{I4};
};

struct ArmatureData {
    uint32_t Version{1}; // Increments when structure changes
    BoneId NextBoneId{1}; // Monotonic allocator state

    std::vector<ArmatureBone> Bones;
    std::unordered_map<BoneId, uint32_t> BoneIdToIndex;

    BoneId AllocateBoneId();
    std::optional<uint32_t> FindBoneIndex(BoneId) const;
    // Returns stable BoneId. Dense indices are runtime/evaluation details and may change across rebuilds.
    BoneId AddBone(std::string_view name = {}, std::optional<BoneId> parent_bone_id = {}, const Transform &rest_local = {});
    void RebuildCaches();
};

// Component on armature object entities in the scene.
// References a shared ArmatureData entity.
struct ArmatureObject {
    entt::entity DataEntity{null_entity};
};
