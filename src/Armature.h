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
    BoneId ParentBoneId{InvalidBoneId}; // Stable structural parent relation
    std::optional<uint32_t> JointNodeIndex{}; // Optional imported glTF joint node binding
    uint32_t ParentIndex{InvalidBoneIndex}, FirstChild{InvalidBoneIndex}, NextSibling{InvalidBoneIndex}; // Dense runtime caches

    std::string Name;
    Transform RestLocal{};
    mat4 RestWorld{I4}, InvRestWorld{I4};
};

struct ArmatureImportedSkin {
    uint32_t SkinIndex{};
    std::optional<uint32_t> SkeletonNodeIndex{};
    std::optional<uint32_t> AnchorNodeIndex{};
    std::vector<uint32_t> OrderedJointNodeIndices;
    std::vector<mat4> InverseBindMatrices;
};

struct ArmatureData {
    uint32_t Version{1}; // Increments when structure changes
    BoneId NextBoneId{1}; // Monotonic allocator state
    bool Dirty{false}; // True when structural edits need finalization.

    std::vector<ArmatureBone> Bones;
    std::unordered_map<BoneId, uint32_t> BoneIdToIndex;
    std::unordered_map<uint32_t, BoneId> JointNodeIndexToBoneId; // Derived cache from ArmatureBone::JointNodeIndex
    std::optional<ArmatureImportedSkin> ImportedSkin{};

    BoneId AllocateBoneId();
    std::optional<uint32_t> FindBoneIndex(BoneId) const;
    std::optional<uint32_t> FindJointNodeIndex(BoneId) const;
    std::optional<BoneId> FindBoneIdByJointNodeIndex(uint32_t) const;
    void SetJointNodeMapping(BoneId, uint32_t joint_node_index);
    // Returns stable BoneId. Dense indices are runtime/evaluation details and may change across rebuilds.
    BoneId AddBone(
        std::string_view name = {},
        std::optional<BoneId> parent_bone_id = {},
        const Transform &rest_local = {},
        std::optional<uint32_t> joint_node_index = {}
    );
    void FinalizeStructure(); // Validate/canonicalize + rebuild derived runtime caches.
};

// Component on armature object entities in the scene.
// References a shared ArmatureData entity.
struct ArmatureObject {
    entt::entity DataEntity{null_entity};
};
