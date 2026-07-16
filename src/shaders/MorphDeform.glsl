// Apply morph target (blend shape) deformation in mesh-local space.
// No-op when draw has no morph data (MorphDeform.Slot == INVALID_SLOT).

// `weights_slot` selects which pose to deform against, so the velocity pass can reach the
// shutter-open and shutter-close poses using the current frame's per-draw offsets.
void ApplyMorphDeform(DrawData draw, inout vec3 position, uint vertex_index, inout vec3 normal, uint weights_slot) {
    if (draw.MorphDeformOffset == INVALID_OFFSET) return;

    bool any_applied = false;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        float weight = MorphWeightBuffers[nonuniformEXT(weights_slot)]
            .Weights[draw.MorphWeightsOffset + t];
        if (weight == 0.0) continue;
        MorphTargetVertex target = MorphTargetBuffers[nonuniformEXT(SceneViewUBO.MorphDeformSlot)]
            .Vertices[draw.MorphDeformOffset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * target.PositionDelta;
        normal += weight * target.NormalDelta;
        any_applied = true;
    }
    if (any_applied) normal = normalize(normal);
}

void ApplyMorphDeform(DrawData draw, inout vec3 position, uint vertex_index, inout vec3 normal) {
    ApplyMorphDeform(draw, position, vertex_index, normal, SceneViewUBO.MorphWeightsSlot);
}
