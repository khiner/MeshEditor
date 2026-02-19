// Apply morph target (blend shape) deformation in mesh-local space.
// No-op when draw has no morph data (MorphDeform.Slot == INVALID_SLOT).

void ApplyMorphDeform(DrawData draw, inout vec3 position, uint vertex_index, inout vec3 normal) {
    if (draw.MorphDeformOffset.Slot == INVALID_SLOT) return;

    bool any_applied = false;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        float weight = MorphWeightBuffers[nonuniformEXT(draw.MorphWeightsOffset.Slot)]
            .Weights[draw.MorphWeightsOffset.Offset + t];
        if (weight == 0.0) continue;
        MorphTargetVertex target = MorphTargetBuffers[nonuniformEXT(draw.MorphDeformOffset.Slot)]
            .Vertices[draw.MorphDeformOffset.Offset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * target.PositionDelta;
        normal += weight * target.NormalDelta;
        any_applied = true;
    }
    if (any_applied) normal = normalize(normal);
}
