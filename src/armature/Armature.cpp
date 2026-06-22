#include "armature/Armature.h"
#include "TransformMath.h"
#include "animation/AnimationData.h"
#include "armature/ArmatureComponents.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"

#include <entt/entity/registry.hpp>

#include <format>

namespace {
void RebuildDerivedCaches(Armature &armature) {
    auto &bones = armature.Bones;
    armature.BoneIdToIndex.clear();
    armature.BoneIdToIndex.reserve(bones.size());

    std::unordered_map<uint32_t, uint32_t> joint_to_bone;
    for (uint32_t i = 0; i < bones.size(); ++i) {
        bones[i].ParentIndex = bones[i].FirstChild = bones[i].NextSibling = InvalidBoneIndex;
        armature.BoneIdToIndex.emplace(bones[i].Id, i);
        if (bones[i].JointNodeIndex) joint_to_bone[*bones[i].JointNodeIndex] = i;
    }
    for (uint32_t i = 0; i < bones.size(); ++i) {
        if (bones[i].ParentBoneId == InvalidBoneId) continue;
        const auto parent = armature.BoneIdToIndex.find(bones[i].ParentBoneId)->second;
        bones[i].ParentIndex = parent;
        bones[i].NextSibling = bones[parent].FirstChild;
        bones[parent].FirstChild = i;
    }

    // Precompute skin joint order -> bone array index (avoids hash lookups in ComputeDeformMatrices).
    armature.JointOrderToBoneIndex.clear();
    if (armature.ImportedSkin) {
        const auto &joints = armature.ImportedSkin->OrderedJointNodeIndices;
        armature.JointOrderToBoneIndex.resize(joints.size(), InvalidBoneIndex);
        for (uint32_t j = 0; j < joints.size(); ++j) {
            if (auto it = joint_to_bone.find(joints[j]); it != joint_to_bone.end()) {
                armature.JointOrderToBoneIndex[j] = it->second;
            }
        }
    }

    armature.RecomputeRestWorld();
}

// Binary search for the keyframe interval containing `t`. Returns the index of the left keyframe.
uint32_t FindKeyframe(const std::vector<float> &times, float t) {
    if (times.size() <= 1 || t <= times.front()) return 0;
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

    // Reparent children of the removed bone to its parent, adjusting RestLocal
    // to preserve world position: new_local = deleted.RestLocal * child.RestLocal.
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

void Armature::FinalizeStructure() {
    if (!Dirty) return;

    ++Version;

    if (Bones.empty()) {
        BoneIdToIndex.clear();
        JointOrderToBoneIndex.clear();
        Dirty = false;
        return;
    }

    RebuildDerivedCaches(*this);

    Dirty = false;
}

void Armature::ResolveAnimationIndices(AnimationClip &clip) const {
    for (auto &channel : clip.Channels) {
        if (channel.TargetBoneId == InvalidBoneId) continue;
        const auto index = FindBoneIndex(channel.TargetBoneId);
        channel.BoneIndex = index.value_or(InvalidBoneIndex);
    }
}

void Armature::RecomputeRestWorld() {
    for (uint32_t i = 0; i < Bones.size(); ++i) {
        const auto local = ToMatrix(Bones[i].RestLocal);
        const auto parent = Bones[i].ParentIndex;
        Bones[i].RestWorld = parent == InvalidBoneIndex ? local : Bones[parent].RestWorld * local;
        Bones[i].InvRestWorld = glm::inverse(Bones[i].RestWorld);
    }
}

void Armature::RecomputeInverseBindMatrices() {
    if (!ImportedSkin) return;
    auto &ibms = ImportedSkin->InverseBindMatrices;
    for (uint32_t j = 0; j < JointOrderToBoneIndex.size() && j < ibms.size(); ++j) {
        const auto bone_index = JointOrderToBoneIndex[j];
        if (bone_index == InvalidBoneIndex || bone_index >= Bones.size()) continue;
        ibms[j] = Bones[bone_index].InvRestWorld;
    }
}

void EvaluateMorphWeights(const MorphWeightClip &clip, float time_seconds, std::span<float> weights) {
    for (const auto &channel : clip.Channels) {
        if (channel.TimesSeconds.empty()) continue;

        const uint32_t target_count = weights.size();
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
        if (channel.BoneIndex >= bone_pose_local.size() || channel.BoneIndex == InvalidBoneIndex) continue;
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

Transform ComposeWithDelta(const Transform &rest, const Transform &delta) {
    return {.P = rest.P + rest.R * delta.P, .R = glm::normalize(rest.R * delta.R), .S = rest.S * delta.S};
}

Transform AbsoluteToDelta(const Transform &rest, const Transform &absolute) {
    const auto inv_r = glm::conjugate(rest.R); // inverse for unit quaternion
    return {.P = inv_r * (absolute.P - rest.P), .R = glm::normalize(inv_r * absolute.R), .S = absolute.S / rest.S};
}

void EvaluateAnimationDeltas(const AnimationClip &clip, float time, std::span<const ArmatureBone> bones, std::span<Transform> deltas) {
    // Evaluate the clip in place, then convert only the components it animated back to delta-from-rest.
    EvaluateAnimation(clip, time, deltas);
    for (const auto &channel : clip.Channels) {
        if (channel.Target == AnimationPath::Weights) continue; // morph weights, not a bone Transform
        if (channel.BoneIndex >= deltas.size() || channel.BoneIndex == InvalidBoneIndex || channel.TimesSeconds.empty()) continue;
        const auto &rest = bones[channel.BoneIndex].RestLocal;
        auto &d = deltas[channel.BoneIndex]; // the animated component currently holds the clip's absolute value
        switch (channel.Target) {
            case AnimationPath::Translation: d.P = glm::conjugate(rest.R) * (d.P - rest.P); break;
            case AnimationPath::Rotation: d.R = glm::normalize(glm::conjugate(rest.R) * d.R); break;
            case AnimationPath::Scale: d.S = d.S / rest.S; break;
            default: break;
        }
    }
}

namespace {
// Zero-roll rotation: maps +Y to `nor` with a well-defined, smooth reference frame.
// Uses Blender's formula (armature.cc vec_roll_to_mat3_normalized) which handles the
// -Y singularity stably, unlike glm::rotation which is discontinuous there.
quat ZeroRollQuat(vec3 nor) {
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
    return glm::quat_cast(m);
}
} // namespace

mat3 BoneVecRollToMat3(vec3 direction, float roll) {
    const vec3 nor = glm::normalize(direction);
    return glm::mat3_cast(glm::angleAxis(roll, nor) * ZeroRollQuat(nor));
}

void BoneMat3ToVecRoll(const mat3 &m, vec3 &direction, float &roll) {
    direction = m[1]; // Y column is the bone axis.
    const vec3 nor = glm::normalize(direction);
    const quat twist = glm::quat_cast(m) * glm::conjugate(ZeroRollQuat(nor));
    roll = 2.f * std::atan2(glm::dot(vec3{twist.x, twist.y, twist.z}, nor), twist.w);
}

// One deform buffer per armature is shared across mesh instances, so we skip Blender's per-instance mesh-to-armature transform.
// A skinned mesh moved off its armature shifts rigidly instead of stretching.
void ComputeDeformMatrices(
    const Armature &data,
    std::span<const mat4> bone_pose_world, std::span<const mat4> inverse_bind_matrices, std::span<mat4> out_deform_matrices
) {
    if (!data.ImportedSkin || data.Bones.empty()) return;

    for (uint32_t j = 0; j < data.JointOrderToBoneIndex.size() && j < out_deform_matrices.size(); ++j) {
        const auto bone_index = data.JointOrderToBoneIndex[j];
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
    const mat4 constrained_local = glm::inverse(parent_pose_world) * (armature_world_inv * effective_target);
    const Transform tl{vec3(constrained_local[3]), glm::normalize(glm::quat_cast(mat3(constrained_local))), pre_local.S};
    if (c.Influence >= 1.f) return tl;
    return {glm::mix(pre_local.P, tl.P, c.Influence), glm::slerp(pre_local.R, tl.R, c.Influence), pre_local.S};
}

float ComputeBoneDisplayScale(const Armature &armature, uint32_t bone_index) {
    static constexpr float MinBoneLength = 0.004f;
    float min_child_dist = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < armature.Bones.size(); ++j) {
        if (armature.Bones[j].ParentIndex == bone_index) {
            const float d = glm::length(vec3{armature.Bones[j].RestWorld[3]} - vec3{armature.Bones[bone_index].RestWorld[3]});
            if (d > MinBoneLength) min_child_dist = std::min(min_child_dist, d);
        }
    }
    if (min_child_dist < std::numeric_limits<float>::max()) return min_child_dist;
    if (armature.Bones[bone_index].ParentIndex != InvalidBoneIndex) {
        return ComputeBoneDisplayScale(armature, armature.Bones[bone_index].ParentIndex);
    }
    return 1.f;
}

std::vector<uint32_t> CollectBonesForDeletion(const entt::registry &r, entt::entity arm_obj_entity) {
    std::vector<uint32_t> to_delete;
    for (const auto e : r.view<const BoneSelection, const BoneIndex>()) {
        if (r.get<const SubElementOf>(e).Parent == arm_obj_entity) to_delete.emplace_back(r.get<const BoneIndex>(e).Index);
    }
    std::sort(to_delete.rbegin(), to_delete.rend());
    return to_delete;
}
