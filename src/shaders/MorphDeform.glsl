// Apply morph target (blend shape) deformation in mesh-local space.
// No-op when draw has no morph data (MorphDeform.Slot == INVALID_SLOT).

vec3 ApplyMorphDeform(DrawData draw, vec3 position, uint vertex_index) {
    if (draw.MorphDeform.Slot == INVALID_SLOT) return position;

    vec3 result = position;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        float weight = MorphWeightBuffers[nonuniformEXT(draw.MorphWeights.Slot)]
            .Weights[draw.MorphWeights.Offset + t];
        if (weight == 0.0) continue;
        vec3 delta = MorphTargetBuffers[nonuniformEXT(draw.MorphDeform.Slot)]
            .Vertices[draw.MorphDeform.Offset + t * draw.VertexCountOrHeadImageSlot + vertex_index].PositionDelta;
        result += weight * delta;
    }
    return result;
}
