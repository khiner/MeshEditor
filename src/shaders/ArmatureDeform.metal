#pragma once
// Applies armature deformation in mesh-local space and returns the input when no deform data exists.
#include "Bindless.metal"

template<typename SetT>
inline float3 ApplyArmatureDeform(const thread SceneT<SetT> &scene, DrawData draw, float3 position, uint vertex_index, thread float3 &normal) {
    if (draw.BoneDeformOffset == InvalidOffset) return position;

    const BoneDeformVertex bd=scene.BoneDeforms(draw.Skin.Slot)[draw.Skin.Offset+vertex_index];
    device const mat4 *matrices = scene.ArmatureDeforms(scene.View.ArmatureDeformSlot);
    const float4 weights = float4(bd.Weights);
    const uint4 joints = uint4(bd.Joints);

    float4x4 deform = float4x4(0.0f);
    deform += weights.x * matrices[draw.ArmatureDeformOffset + joints.x].Unpack();
    deform += weights.y * matrices[draw.ArmatureDeformOffset + joints.y].Unpack();
    deform += weights.z * matrices[draw.ArmatureDeformOffset + joints.z].Unpack();
    deform += weights.w * matrices[draw.ArmatureDeformOffset + joints.w].Unpack();

    normal = normalize(float3x3(deform[0].xyz, deform[1].xyz, deform[2].xyz) * normal);
    return (deform * float4(position, 1.0f)).xyz;
}
