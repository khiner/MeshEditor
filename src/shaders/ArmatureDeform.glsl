// Deform position + normal in mesh-local space using armature deform matrices.
// No-op when draw has no deform data (BoneDeform.Slot == INVALID_SLOT).

// `deform_slot` selects which pose to deform against, so the velocity pass can reach the
// shutter-open and shutter-close poses using the current frame's per-draw offsets.
vec3 ApplyArmatureDeform(DrawData draw, vec3 position, uint vertex_index, inout vec3 normal, uint deform_slot) {
    if (draw.BoneDeformOffset == INVALID_OFFSET) return position;

    const BoneDeformVertex bd = BoneDeformBuffers[nonuniformEXT(SceneViewUBO.BoneDeformSlot)]
        .Vertices[draw.BoneDeformOffset + vertex_index];

    mat4 deform = mat4(0.0);
    deform += bd.Weights.x * ArmatureDeformBuffers[nonuniformEXT(deform_slot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.x];
    deform += bd.Weights.y * ArmatureDeformBuffers[nonuniformEXT(deform_slot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.y];
    deform += bd.Weights.z * ArmatureDeformBuffers[nonuniformEXT(deform_slot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.z];
    deform += bd.Weights.w * ArmatureDeformBuffers[nonuniformEXT(deform_slot)]
        .Matrices[draw.ArmatureDeformOffset + bd.Joints.w];

    normal = normalize(mat3(deform) * normal);
    return vec3(deform * vec4(position, 1.0));
}

vec3 ApplyArmatureDeform(DrawData draw, vec3 position, uint vertex_index, inout vec3 normal) {
    return ApplyArmatureDeform(draw, position, vertex_index, normal, SceneViewUBO.ArmatureDeformSlot);
}
