// Deform position + normal in mesh-local space using armature deform matrices.
// No-op when draw has no deform data (BoneDeform.Slot == INVALID_SLOT).

vec3 ApplyArmatureDeform(DrawData draw, vec3 position, uint vertex_index, inout vec3 normal) {
    if (draw.BoneDeformOffset.Slot == INVALID_SLOT) return position;

    const BoneDeformVertex bd = BoneDeformBuffers[nonuniformEXT(draw.BoneDeformOffset.Slot)]
        .Vertices[draw.BoneDeformOffset.Offset + vertex_index];

    mat4 deform = mat4(0.0);
    deform += bd.Weights.x * ArmatureDeformBuffers[nonuniformEXT(draw.ArmatureDeformOffset.Slot)]
        .Matrices[draw.ArmatureDeformOffset.Offset + bd.Joints.x];
    deform += bd.Weights.y * ArmatureDeformBuffers[nonuniformEXT(draw.ArmatureDeformOffset.Slot)]
        .Matrices[draw.ArmatureDeformOffset.Offset + bd.Joints.y];
    deform += bd.Weights.z * ArmatureDeformBuffers[nonuniformEXT(draw.ArmatureDeformOffset.Slot)]
        .Matrices[draw.ArmatureDeformOffset.Offset + bd.Joints.z];
    deform += bd.Weights.w * ArmatureDeformBuffers[nonuniformEXT(draw.ArmatureDeformOffset.Slot)]
        .Matrices[draw.ArmatureDeformOffset.Offset + bd.Joints.w];

    normal = normalize(mat3(deform) * normal);
    return vec3(deform * vec4(position, 1.0));
}
