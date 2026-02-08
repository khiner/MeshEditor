#include "Armature.h"

#include <format>
#include <stdexcept>

namespace {
mat4 ToMatrix(const Transform &t) { return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S); }
} // namespace

BoneId ArmatureData::AllocateBoneId() {
    if (NextBoneId == InvalidBoneId) throw std::runtime_error{"Armature bone ID allocator overflowed."};
    return NextBoneId++;
}

std::optional<uint32_t> ArmatureData::FindBoneIndex(BoneId bone_id) const {
    if (bone_id == InvalidBoneId) return std::nullopt;
    if (const auto it = BoneIdToIndex.find(bone_id); it != BoneIdToIndex.end()) return it->second;
    return std::nullopt;
}

BoneId ArmatureData::AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local) {
    std::optional<uint32_t> parent_index;
    if (parent_bone_id) {
        if (*parent_bone_id == InvalidBoneId) throw std::invalid_argument{"Invalid parent bone ID (InvalidBoneId)."};

        parent_index = FindBoneIndex(*parent_bone_id);
        if (!parent_index) throw std::out_of_range{std::format("Parent bone ID {} does not exist.", *parent_bone_id)};
    }

    ArmatureBone bone;
    const BoneId bone_id = AllocateBoneId();
    bone.Id = bone_id;
    bone.ParentIndex = parent_index.value_or(InvalidBoneIndex);
    bone.Name = name.empty() ? std::format("Bone{}", Bones.size()) : std::string(name);
    bone.RestLocal = rest_local;

    Bones.emplace_back(std::move(bone));
    RebuildCaches();
    return bone_id;
}

void ArmatureData::RebuildCaches() {
    ++Version;

    BoneIdToIndex.clear();
    BoneIdToIndex.reserve(Bones.size());

    for (auto &bone : Bones) {
        bone.FirstChild = InvalidBoneIndex;
        bone.NextSibling = InvalidBoneIndex;
    }

    // Build stable-id mapping first so all tooling can resolve IDs after structure edits.
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto id = Bones[i].Id;
        if (id == InvalidBoneId) throw std::runtime_error{std::format("Bone {} has invalid ID 0.", i)};
        if (!BoneIdToIndex.emplace(id, i).second) throw std::runtime_error{std::format("Duplicate bone ID {}.", id)};
    }

    // Parent-before-child order is required for a single linear evaluation pass.
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto parent = Bones[i].ParentIndex;
        if (parent == InvalidBoneIndex) continue;
        if (parent >= Bones.size()) throw std::runtime_error{std::format("Bone {} references invalid parent index {}.", i, parent)};
        if (parent >= i) throw std::runtime_error{std::format("Bone {} parent index {} violates parent-before-child ordering.", i, parent)};

        Bones[i].NextSibling = Bones[parent].FirstChild;
        Bones[parent].FirstChild = i;
    }

    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto local = ToMatrix(Bones[i].RestLocal);
        const auto parent = Bones[i].ParentIndex;
        Bones[i].RestWorld = parent == InvalidBoneIndex ? local : Bones[parent].RestWorld * local;
        Bones[i].InvRestWorld = glm::inverse(Bones[i].RestWorld);
    }
}
