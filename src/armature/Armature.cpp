#include "armature/Armature.h"
#include "TransformMath.h"
#include "armature/ArmatureComponents.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"

#include "state/Scene.h"

#include <format>

BoneId Armature::AllocateBoneId() {
    if (NextBoneId == InvalidBoneId) throw std::runtime_error{"Armature bone ID allocator overflowed."};
    return NextBoneId++;
}

std::optional<uint32_t> Armature::FindBoneIndex(BoneId bone_id) const {
    if (bone_id == InvalidBoneId) return {};
    if (const auto it = BoneIdToIndex.find(bone_id); it != BoneIdToIndex.end()) return it->second;
    return {};
}

BoneId Armature::AddBone(std::string_view name, std::optional<BoneId> parent_bone_id, const Transform &rest_local, std::optional<uint32_t> joint_node_index) {
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

bool Armature::RemoveBone(BoneId bone_id) {
    const auto it = BoneIdToIndex.find(bone_id);
    if (it == BoneIdToIndex.end()) return false;

    const auto index = it->second;

    // Preserve each child's world rest pose while reparenting it to the removed bone's parent.
    const auto parent_id = Bones[index].ParentBoneId;
    const auto &deleted_rest = Bones[index].RestLocal;
    for (auto &bone : Bones) {
        if (bone.ParentBoneId == bone_id) {
            bone.ParentBoneId = parent_id;
            bone.RestLocal = ComposeLocalTransforms(deleted_rest, bone.RestLocal);
        }
    }

    Bones.erase(Bones.begin() + index);

    BoneIdToIndex.clear();
    for (uint32_t i = 0; i < Bones.size(); ++i) BoneIdToIndex[Bones[i].Id] = i;

    Dirty = true;
    return true;
}

void Armature::RebuildCaches() {
    BoneIdToIndex.clear();
    BoneIdToIndex.reserve(Bones.size());

    std::unordered_map<uint32_t, uint32_t> joint_to_bone;
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        Bones[i].ParentIndex = Bones[i].FirstChild = Bones[i].NextSibling = InvalidBoneIndex;
        BoneIdToIndex.emplace(Bones[i].Id, i);
        if (Bones[i].JointNodeIndex) joint_to_bone[*Bones[i].JointNodeIndex] = i;
    }
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        if (Bones[i].ParentBoneId == InvalidBoneId) continue;
        const auto parent = BoneIdToIndex.find(Bones[i].ParentBoneId)->second;
        Bones[i].ParentIndex = parent;
        Bones[i].NextSibling = Bones[parent].FirstChild;
        Bones[parent].FirstChild = i;
    }

    JointOrderToBoneIndex.assign(Skins.size(), {});
    for (uint32_t s = 0; s < Skins.size(); ++s) {
        const auto &joints = Skins[s].OrderedJointNodeIndices;
        auto &joint_map = JointOrderToBoneIndex[s];
        joint_map.resize(joints.size(), InvalidBoneIndex);
        for (uint32_t j = 0; j < joints.size(); ++j) {
            if (auto it = joint_to_bone.find(joints[j]); it != joint_to_bone.end()) {
                joint_map[j] = it->second;
            }
        }
    }

    RecomputeRestWorld();
}

void Armature::FinalizeStructure() {
    if (!Dirty) return;

    ++Version;

    if (Bones.empty()) {
        BoneIdToIndex.clear();
        JointOrderToBoneIndex.clear();
        Dirty = false;
        return;
    }

    RebuildCaches();

    Dirty = false;
}

void Armature::RecomputeRestWorld() {
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto local = ToMatrix(Bones[i].RestLocal);
        const auto parent = Bones[i].ParentIndex;
        Bones[i].RestWorld = parent == InvalidBoneIndex ? local : Bones[parent].RestWorld * local;
        Bones[i].InvRestWorld = numeric::Inverse(Bones[i].RestWorld);
    }
}

void Armature::RecomputeInverseBindMatrices() {
    for (uint32_t s = 0; s < Skins.size() && s < JointOrderToBoneIndex.size(); ++s) {
        auto &ibms = Skins[s].InverseBindMatrices;
        const auto &joint_map = JointOrderToBoneIndex[s];
        for (uint32_t j = 0; j < joint_map.size() && j < ibms.size(); ++j) {
            const auto bone_index = joint_map[j];
            if (bone_index != InvalidBoneIndex && bone_index < Bones.size()) {
                ibms[j] = Bones[bone_index].InvRestWorld;
            }
        }
    }
}

Transform ComposeWithDelta(const Transform &rest, const Transform &delta) {
    return {.P = rest.P + rest.R * delta.P, .R = numeric::Normalize(rest.R * delta.R), .S = rest.S * delta.S};
}

Transform AbsoluteToDelta(const Transform &rest, const Transform &absolute) {
    const auto inv_r = numeric::Conjugate(rest.R);
    return {.P = inv_r * (absolute.P - rest.P), .R = numeric::Normalize(inv_r * absolute.R), .S = absolute.S / rest.S};
}

namespace {
quat ZeroRollQuat(vec3 nor) {
    // Blender's armature.cc vec_roll_to_mat3_normalized formula is continuous at the -Y singularity.
    const float x = nor.x, y = nor.y, z = nor.z;
    constexpr float SafeThreshold = 6.1e-3f, CriticalThresholdSq = 2.5e-4f * 2.5e-4f;
    const float theta = 1.f + y;
    const float theta_alt = x * x + z * z;

    mat3 m;
    if (theta > SafeThreshold || theta_alt > CriticalThresholdSq) {
        const float t = (theta <= SafeThreshold) ? theta_alt * 0.5f + theta_alt * theta_alt * 0.125f : theta;
        m[0] = {1 - x * x / t, -x, -x * z / t};
        m[1] = {x, y, z};
        m[2] = {-x * z / t, -z, 1 - z * z / t};
    } else {
        m = {-1, 0, 0, 0, -1, 0, 0, 0, 1};
    }
    return numeric::ToQuat(m);
}
} // namespace

mat3 BoneVecRollToMat3(vec3 direction, float roll) {
    const vec3 nor = numeric::Normalize(direction);
    return numeric::ToMat3(numeric::AngleAxis(roll, nor) * ZeroRollQuat(nor));
}

void BoneMat3ToVecRoll(const mat3 &m, vec3 &direction, float &roll) {
    direction = m[1];
    const vec3 nor = numeric::Normalize(direction);
    const quat twist = numeric::ToQuat(m) * numeric::Conjugate(ZeroRollQuat(nor));
    roll = 2.f * std::atan2(numeric::Dot(vec3{twist.x, twist.y, twist.z}, nor), twist.w);
}

// One deform buffer per skin is shared across mesh instances, so a skinned mesh moved off its armature shifts rigidly instead of stretching.
void ComputeDeformMatrices(
    const Armature &data, uint32_t skin_slot,
    std::span<const mat4> bone_pose_world, std::span<mat4> out_deform_matrices
) {
    if (skin_slot >= data.Skins.size() || skin_slot >= data.JointOrderToBoneIndex.size() || data.Bones.empty()) return;

    const auto &inverse_bind_matrices = data.Skins[skin_slot].InverseBindMatrices;
    const auto &joint_map = data.JointOrderToBoneIndex[skin_slot];
    for (uint32_t j = 0; j < joint_map.size() && j < out_deform_matrices.size(); ++j) {
        const auto bone_index = joint_map[j];
        if (bone_index == InvalidBoneIndex || bone_index >= bone_pose_world.size()) {
            out_deform_matrices[j] = I4;
            continue;
        }
        const auto &ibm = (j < inverse_bind_matrices.size()) ? inverse_bind_matrices[j] : I4;
        out_deform_matrices[j] = bone_pose_world[bone_index] * ibm;
    }
}

Transform ApplyBoneConstraint(
    const BoneConstraint &c, const Transform &pre_local,
    const mat4 &parent_pose_world, const mat4 &armature_world_inv, const mat4 &target_world
) {
    const mat4 effective_target = std::visit(
        [&]<typename T>(const T &d) -> mat4 {
            if constexpr (std::is_same_v<T, ChildOfData>) return target_world * d.InverseMatrix;
            else return target_world;
        },
        c.Data
    );
    const mat4 constrained_local = numeric::Inverse(parent_pose_world) * (armature_world_inv * effective_target);
    const Transform tl{vec3(constrained_local[3]), numeric::Normalize(numeric::ToQuat(mat3(constrained_local))), pre_local.S};
    if (c.Influence >= 1.f) return tl;
    return {numeric::Mix(pre_local.P, tl.P, c.Influence), numeric::Slerp(pre_local.R, tl.R, c.Influence), pre_local.S};
}

float ComputeBoneDisplayScale(const Armature &armature, uint32_t bone_index) {
    static constexpr float MinBoneLength = 0.004f;
    float min_child_dist = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < armature.Bones.size(); ++j) {
        if (armature.Bones[j].ParentIndex == bone_index) {
            const float d = numeric::Length(vec3{armature.Bones[j].RestWorld[3]} - vec3{armature.Bones[bone_index].RestWorld[3]});
            if (d > MinBoneLength) min_child_dist = std::min(min_child_dist, d);
        }
    }
    if (min_child_dist < std::numeric_limits<float>::max()) return min_child_dist;
    if (armature.Bones[bone_index].ParentIndex != InvalidBoneIndex) {
        return ComputeBoneDisplayScale(armature, armature.Bones[bone_index].ParentIndex);
    }
    return 1.f;
}

std::vector<uint32_t> CollectBonesForDeletion(const state::Scene &r, state::Entity arm_obj_entity) {
    std::vector<uint32_t> to_delete;
    for (const auto e : r.view<const BoneSelection, const BoneIndex>()) {
        if (r.get<const SubElementOf>(e).Parent == arm_obj_entity) to_delete.emplace_back(r.get<const BoneIndex>(e).Index);
    }
    std::sort(to_delete.rbegin(), to_delete.rend());
    return to_delete;
}
