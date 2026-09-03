#ifndef ARMATURE_DEFORM_MSL
#define ARMATURE_DEFORM_MSL

// Applies armature deformation in mesh-local space and returns the input when no deform data exists.
#include "Bindless.metal"

// `deform_slot` selects current, shutter-open, or shutter-close pose matrices.
template<typename SetT>
inline float3 ApplyArmatureDeform(const thread SceneT<SetT> &scene, DrawData draw, float3 position, uint vertex_index, thread float3 &normal, uint deform_slot) {
    if (draw.BoneDeformOffset == INVALID_OFFSET) return position;

    const BoneDeformVertex bd = scene.BoneDeforms(scene.View.BoneDeformSlot)[draw.BoneDeformOffset + vertex_index];
    device const packed_float4x4 *matrices = scene.ArmatureDeforms(deform_slot);
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

template<typename SetT>
inline float3 ApplyArmatureDeform(const thread SceneT<SetT> &scene, DrawData draw, float3 position, uint vertex_index, thread float3 &normal) {
    return ApplyArmatureDeform(scene, draw, position, vertex_index, normal, scene.View.ArmatureDeformSlot);
}

#endif
