// Apply morph target (blend shape) deformation in mesh-local space.
// No-op when draw has no morph data (MorphDeform.Slot == INVALID_SLOT).
// Positions only: shading normals rederive from the morphed positions.

// `weights_slot` selects which pose to deform against, so the velocity pass can reach the
// shutter-open and shutter-close poses using the current frame's per-draw offsets.
void ApplyMorphDeform(DrawData draw, inout vec3 position, uint vertex_index, uint weights_slot) {
    if (draw.MorphDeformOffset == INVALID_OFFSET) return;

    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        float weight = MorphWeightBuffers[nonuniformEXT(weights_slot)]
            .Weights[draw.MorphWeightsOffset + t];
        if (weight == 0.0) continue;
        MorphTargetVertex target = MorphTargetBuffers[nonuniformEXT(SceneViewUBO.MorphDeformSlot)]
            .Vertices[draw.MorphDeformOffset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * target.PositionDelta;
    }
}

// Like ApplyMorphDeform, fetching each target vertex once.
// Authored-morph draws also accumulate the weighted authored normal deltas.
void ApplyMorphDeform(DrawData draw, inout vec3 position, inout vec3 normal_delta, uint vertex_index) {
    if (draw.MorphDeformOffset == INVALID_OFFSET) return;

    const bool authored = draw.MorphShadingAuthored != 0u;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        float weight = MorphWeightBuffers[nonuniformEXT(SceneViewUBO.MorphWeightsSlot)]
            .Weights[draw.MorphWeightsOffset + t];
        if (weight == 0.0) continue;
        MorphTargetVertex target = MorphTargetBuffers[nonuniformEXT(SceneViewUBO.MorphDeformSlot)]
            .Vertices[draw.MorphDeformOffset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * target.PositionDelta;
        if (authored) normal_delta += weight * target.NormalDelta;
    }
}
