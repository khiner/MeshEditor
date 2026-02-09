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
    if (bone_id == InvalidBoneId) return {};
    if (const auto it = BoneIdToIndex.find(bone_id); it != BoneIdToIndex.end()) return it->second;
    return {};
}

std::optional<uint32_t> ArmatureData::FindJointNodeIndex(BoneId bone_id) const {
    const auto index = FindBoneIndex(bone_id);
    if (!index) return {};
    return Bones[*index].JointNodeIndex;
}

std::optional<BoneId> ArmatureData::FindBoneIdByJointNodeIndex(uint32_t joint_node_index) const {
    if (!Dirty) {
        if (const auto it = JointNodeIndexToBoneId.find(joint_node_index); it != JointNodeIndexToBoneId.end()) return it->second;
        return {};
    }

    for (const auto &bone : Bones) {
        if (bone.JointNodeIndex && *bone.JointNodeIndex == joint_node_index) return bone.Id;
    }
    return {};
}

void ArmatureData::SetJointNodeMapping(BoneId bone_id, uint32_t joint_node_index) {
    const auto index = FindBoneIndex(bone_id);
    if (!index) throw std::out_of_range{std::format("Bone ID {} does not exist.", bone_id)};
    Bones[*index].JointNodeIndex = joint_node_index;
    Dirty = true;
}

BoneId ArmatureData::AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local, std::optional<uint32_t> joint_node_index) {
    if (parent_bone_id) {
        if (*parent_bone_id == InvalidBoneId) throw std::invalid_argument{"Invalid parent bone ID (InvalidBoneId)."};
        if (!FindBoneIndex(*parent_bone_id)) throw std::out_of_range{std::format("Parent bone ID {} does not exist.", *parent_bone_id)};
    }

    const auto bone_id = AllocateBoneId();
    Bones.emplace_back(ArmatureBone{
        .Id = bone_id,
        .ParentBoneId = parent_bone_id.value_or(InvalidBoneId),
        .JointNodeIndex = joint_node_index,
        .Name = name.empty() ? std::format("Bone{}", Bones.size()) : std::string(name),
        .RestLocal = rest_local,
    });
    BoneIdToIndex[bone_id] = Bones.size() - 1;
    Dirty = true;
    return bone_id;
}

void ArmatureData::FinalizeStructure() {
    if (!Dirty) return;

    ++Version;

    if (Bones.empty()) {
        BoneIdToIndex.clear();
        JointNodeIndexToBoneId.clear();
        Dirty = false;
        return;
    }

    std::unordered_map<BoneId, uint32_t> old_index_by_id;
    old_index_by_id.reserve(Bones.size());
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto id = Bones[i].Id;
        if (id == InvalidBoneId) throw std::runtime_error{std::format("Bone {} has invalid ID 0.", i)};
        if (const auto [existing_it, inserted] = old_index_by_id.emplace(id, i); !inserted) {
            throw std::runtime_error{std::format("Duplicate bone ID {} at indices {} and {}.", id, existing_it->second, i)};
        }
    }

    std::vector<std::vector<uint32_t>> children_by_old_index(Bones.size());
    std::vector<uint32_t> indegree(Bones.size(), 0);
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto parent_id = Bones[i].ParentBoneId;
        if (parent_id == InvalidBoneId) continue;
        if (parent_id == Bones[i].Id) throw std::runtime_error{std::format("Bone ID {} cannot parent itself.", parent_id)};

        const auto parent_it = old_index_by_id.find(parent_id);
        if (parent_it == old_index_by_id.end()) throw std::runtime_error{std::format("Bone {} references missing parent ID {}.", Bones[i].Id, parent_id)};

        children_by_old_index[parent_it->second].push_back(i);
        indegree[i] = 1;
    }

    std::vector<uint32_t> ordered_old_indices;
    ordered_old_indices.reserve(Bones.size());
    std::vector<uint32_t> queue;
    queue.reserve(Bones.size());
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        if (indegree[i] == 0) queue.push_back(i);
    }

    uint32_t queue_read = 0;
    while (queue_read < queue.size()) {
        const auto current = queue[queue_read++];
        ordered_old_indices.push_back(current);
        for (const auto child : children_by_old_index[current]) {
            if (--indegree[child] == 0) queue.push_back(child);
        }
    }
    if (ordered_old_indices.size() != Bones.size()) throw std::runtime_error{"Armature has cyclic or invalid parent relations."};

    std::vector<ArmatureBone> reordered;
    reordered.reserve(Bones.size());
    for (const auto old_index : ordered_old_indices) reordered.emplace_back(std::move(Bones[old_index]));
    Bones = std::move(reordered);

    BoneIdToIndex.clear();
    BoneIdToIndex.reserve(Bones.size());
    JointNodeIndexToBoneId.clear();
    JointNodeIndexToBoneId.reserve(Bones.size());

    for (auto &bone : Bones) bone.ParentIndex = bone.FirstChild = bone.NextSibling = InvalidBoneIndex;

    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto id = Bones[i].Id;
        if (id == InvalidBoneId) throw std::runtime_error{std::format("Bone {} has invalid ID 0.", i)};
        if (const auto [existing_it, inserted] = BoneIdToIndex.emplace(id, i); !inserted) {
            throw std::runtime_error{std::format("Duplicate bone ID {} at indices {} and {}.", id, existing_it->second, i)};
        }
    }

    for (uint32_t i = 0; i < Bones.size(); ++i) {
        if (const auto joint_node_index = Bones[i].JointNodeIndex) {
            if (const auto [existing_it, inserted] = JointNodeIndexToBoneId.emplace(*joint_node_index, Bones[i].Id); !inserted) {
                throw std::runtime_error{std::format("Duplicate joint node index {} mapped to bone IDs {} and {}.", *joint_node_index, existing_it->second, Bones[i].Id)};
            }
        }
    }

    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto parent_id = Bones[i].ParentBoneId;
        if (parent_id == InvalidBoneId) continue;

        const auto parent_it = BoneIdToIndex.find(parent_id);
        if (parent_it == BoneIdToIndex.end()) throw std::runtime_error{std::format("Bone {} references missing parent ID {}.", Bones[i].Id, parent_id)};

        const auto parent = parent_it->second;
        if (parent >= i) throw std::runtime_error{std::format("Bone {} parent ID {} is not ordered before child after rebuild.", Bones[i].Id, parent_id)};

        Bones[i].ParentIndex = parent;
        Bones[i].NextSibling = Bones[parent].FirstChild;
        Bones[parent].FirstChild = i;
    }

    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto local = ToMatrix(Bones[i].RestLocal);
        const auto parent = Bones[i].ParentIndex;
        Bones[i].RestWorld = parent == InvalidBoneIndex ? local : Bones[parent].RestWorld * local;
        Bones[i].InvRestWorld = glm::inverse(Bones[i].RestWorld);
    }

    Dirty = false;
}
