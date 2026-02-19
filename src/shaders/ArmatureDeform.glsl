// Deform position + normal in mesh-local space using armature deform matrices.
// No-op when draw has no deform data (BoneDeform.Slot == INVALID_SLOT).

vec3 ApplyArmatureDeform(DrawData draw, vec3 position, uint vertex_index, inout vec3 normal) {
    if (draw.BoneDeformOffset == INVALID_OFFSET) return position;

    const BoneDeformVertex bd = BoneDeformBuffers[nonuniformEXT(SceneViewUBO.BoneDeformSlot)]
        .Vertices[draw.BoneDeformOffset + vertex_index];

    mat4 deform = mat4(0.0);
    deform += bd.Weights.x * ArmatureDeformBuffers[nonuniformEXT(SceneViewUBO.ArmatureDeformSlot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.x];
    deform += bd.Weights.y * ArmatureDeformBuffers[nonuniformEXT(SceneViewUBO.ArmatureDeformSlot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.y];
    deform += bd.Weights.z * ArmatureDeformBuffers[nonuniformEXT(SceneViewUBO.ArmatureDeformSlot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.z];
    deform += bd.Weights.w * ArmatureDeformBuffers[nonuniformEXT(SceneViewUBO.ArmatureDeformSlot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.w];

    normal = normalize(mat3(deform) * normal);
    return vec3(deform * vec4(position, 1.0));
}
