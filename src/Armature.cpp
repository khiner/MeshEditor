#include "Armature.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace {
mat4 ToMatrix(const Transform &t) { return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S); }

// Binary search for the keyframe interval containing `t`. Returns the index of the left keyframe.
uint32_t FindKeyframe(const std::vector<float> &times, float t) {
    if (times.size() <= 1) return 0;
    if (t <= times.front()) return 0;
    if (t >= times.back()) return times.size() - 2;
    auto it = std::upper_bound(times.begin(), times.end(), t);
    if (it == times.begin()) return 0;
    return std::distance(times.begin(), it) - 1;
}

vec3 ReadVec3(const float *data) { return {data[0], data[1], data[2]}; }
quat ReadQuat(const float *data) { return {data[3], data[0], data[1], data[2]}; } // glTF: xyzw -> glm: wxyz

// Cubic Hermite interpolation
vec3 CubicHermite(vec3 p0, vec3 m0, vec3 p1, vec3 m1, float t) {
    const float t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1;
}

quat CubicHermiteQuat(quat p0, quat m0, quat p1, quat m1, float t) {
    const float t2 = t * t, t3 = t2 * t;
    const quat result = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1;
    return glm::normalize(result);
}
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

void EvaluateMorphWeights(const MorphWeightClip &clip, float time_seconds, std::span<float> weights) {
    for (const auto &channel : clip.Channels) {
        if (channel.TimesSeconds.empty()) continue;

        const auto target_count = static_cast<uint32_t>(weights.size());
        if (target_count == 0) continue;

        const auto k = FindKeyframe(channel.TimesSeconds, time_seconds);
        const uint32_t last = channel.TimesSeconds.size() - 1;
        const uint32_t k1 = std::min(k + 1, last);

        if (channel.Interp == AnimationInterpolation::Step) {
            const float *v = channel.Values.data() + k * target_count;
            for (uint32_t t = 0; t < target_count; ++t) weights[t] = v[t];
        } else if (channel.Interp == AnimationInterpolation::Linear) {
            const float t0 = channel.TimesSeconds[k], t1 = channel.TimesSeconds[k1];
            const float alpha = (t1 > t0) ? std::clamp((time_seconds - t0) / (t1 - t0), 0.f, 1.f) : 0.f;
            const float *v0 = channel.Values.data() + k * target_count;
            const float *v1 = channel.Values.data() + k1 * target_count;
            for (uint32_t t = 0; t < target_count; ++t) weights[t] = glm::mix(v0[t], v1[t], alpha);
        } else { // CubicSpline
            const float t0 = channel.TimesSeconds[k], t1 = channel.TimesSeconds[k1];
            const float dt = t1 - t0;
            const float alpha = (dt > 0) ? std::clamp((time_seconds - t0) / dt, 0.f, 1.f) : 0.f;
            const float t2 = alpha * alpha, t3 = t2 * alpha;
            // CubicSpline: each keyframe stores [in-tangent, value, out-tangent], each of size target_count
            const uint32_t stride = target_count * 3;
            const float *kf0 = channel.Values.data() + k * stride;
            const float *kf1 = channel.Values.data() + k1 * stride;
            for (uint32_t tw = 0; tw < target_count; ++tw) {
                const float val0 = kf0[target_count + tw];
                const float out0 = kf0[2 * target_count + tw];
                const float in1 = kf1[tw];
                const float val1 = kf1[target_count + tw];
                weights[tw] = (2 * t3 - 3 * t2 + 1) * val0 + (t3 - 2 * t2 + alpha) * dt * out0 +
                    (-2 * t3 + 3 * t2) * val1 + (t3 - t2) * dt * in1;
            }
        }
    }
}

void EvaluateAnimation(const AnimationClip &clip, float time_seconds, std::span<Transform> bone_pose_local) {
    for (const auto &channel : clip.Channels) {
        if (channel.Target == AnimationPath::Weights) continue; // Handled by EvaluateMorphWeights
        if (channel.BoneIndex >= bone_pose_local.size()) continue;
        if (channel.TimesSeconds.empty()) continue;

        const auto k = FindKeyframe(channel.TimesSeconds, time_seconds);
        auto &pose = bone_pose_local[channel.BoneIndex];

        const bool is_rotation = channel.Target == AnimationPath::Rotation;
        const uint32_t comp = is_rotation ? 4 : 3;

        const uint32_t last = channel.TimesSeconds.size() - 1;
        const uint32_t k1 = std::min(k + 1, last);

        if (channel.Interp == AnimationInterpolation::Step) {
            const float *v = channel.Values.data() + k * comp;
            switch (channel.Target) {
                case AnimationPath::Translation: pose.P = ReadVec3(v); break;
                case AnimationPath::Rotation: pose.R = ReadQuat(v); break;
                case AnimationPath::Scale: pose.S = ReadVec3(v); break;
                default: break;
            }
        } else if (channel.Interp == AnimationInterpolation::Linear) {
            const float t0 = channel.TimesSeconds[k], t1 = channel.TimesSeconds[k1];
            const float alpha = (t1 > t0) ? std::clamp((time_seconds - t0) / (t1 - t0), 0.f, 1.f) : 0.f;
            const float *v0 = channel.Values.data() + k * comp;
            const float *v1 = channel.Values.data() + k1 * comp;
            switch (channel.Target) {
                case AnimationPath::Translation: pose.P = glm::mix(ReadVec3(v0), ReadVec3(v1), alpha); break;
                case AnimationPath::Rotation: pose.R = glm::slerp(ReadQuat(v0), ReadQuat(v1), alpha); break;
                case AnimationPath::Scale: pose.S = glm::mix(ReadVec3(v0), ReadVec3(v1), alpha); break;
                default: break;
            }
        } else { // CubicSpline
            const float t0 = channel.TimesSeconds[k], t1 = channel.TimesSeconds[k1];
            const float dt = t1 - t0;
            const float alpha = (dt > 0) ? std::clamp((time_seconds - t0) / dt, 0.f, 1.f) : 0.f;
            // CubicSpline: each keyframe stores [in-tangent, value, out-tangent], each of size `comp`
            const uint32_t stride = comp * 3;
            const float *kf0 = channel.Values.data() + k * stride;
            const float *kf1 = channel.Values.data() + k1 * stride;
            // in0 = kf0[0..comp-1], val0 = kf0[comp..2*comp-1], out0 = kf0[2*comp..3*comp-1]
            const float *val0 = kf0 + comp;
            const float *out0 = kf0 + 2 * comp;
            const float *in1 = kf1;
            const float *val1 = kf1 + comp;
            switch (channel.Target) {
                case AnimationPath::Translation:
                    pose.P = CubicHermite(ReadVec3(val0), dt * ReadVec3(out0), ReadVec3(val1), dt * ReadVec3(in1), alpha);
                    break;
                case AnimationPath::Rotation:
                    pose.R = CubicHermiteQuat(ReadQuat(val0), dt * ReadQuat(out0), ReadQuat(val1), dt * ReadQuat(in1), alpha);
                    break;
                case AnimationPath::Scale:
                    pose.S = CubicHermite(ReadVec3(val0), dt * ReadVec3(out0), ReadVec3(val1), dt * ReadVec3(in1), alpha);
                    break;
                default: break;
            }
        }
    }
}

void ComputeDeformMatrices(
    const ArmatureData &data,
    std::span<const Transform> bone_pose_local, std::span<const mat4> inverse_bind_matrices, std::span<mat4> out_deform_matrices
) {
    if (!data.ImportedSkin || data.Bones.empty()) return;

    // Compute posed world transforms in parent-before-child order (bones are already sorted this way)
    std::vector<mat4> pose_world(data.Bones.size());
    for (uint32_t i = 0; i < data.Bones.size(); ++i) {
        const auto local = ToMatrix(bone_pose_local[i]);
        const auto parent = data.Bones[i].ParentIndex;
        pose_world[i] = (parent == InvalidBoneIndex) ? local : pose_world[parent] * local;
    }

    // For each joint in the skin's ordering, compute the deform matrix.
    const auto &ordered_joints = data.ImportedSkin->OrderedJointNodeIndices;
    for (uint32_t j = 0; j < ordered_joints.size() && j < out_deform_matrices.size(); ++j) {
        const auto joint_node_index = ordered_joints[j];
        const auto bone_id = data.FindBoneIdByJointNodeIndex(joint_node_index);
        if (!bone_id) {
            out_deform_matrices[j] = I4;
            continue;
        }
        const auto bone_index = data.FindBoneIndex(*bone_id);
        if (!bone_index || *bone_index >= pose_world.size()) {
            out_deform_matrices[j] = I4;
            continue;
        }
        const auto &ibm = (j < inverse_bind_matrices.size()) ? inverse_bind_matrices[j] : I4;
        out_deform_matrices[j] = pose_world[*bone_index] * ibm;
    }
}
