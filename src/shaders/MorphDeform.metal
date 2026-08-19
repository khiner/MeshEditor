#ifndef MORPH_DEFORM_MSL
#define MORPH_DEFORM_MSL

// Apply morph target (blend shape) deformation in mesh-local space.
// A no-op when the draw has no morph data.
// Positions only: shading normals rederive from the morphed positions.
#include "Bindless.metal"

// `weights_slot` selects which pose to deform against, so the velocity pass can reach the
// shutter-open and shutter-close poses using the current frame's per-draw offsets.
template<typename SetT>
inline void ApplyMorphDeform(const thread SceneT<SetT> &scene, DrawData draw, thread float3 &position, uint vertex_index, uint weights_slot) {
    if (draw.MorphDeformOffset == INVALID_OFFSET) return;

    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        const float weight = scene.MorphWeights(weights_slot)[draw.MorphWeightsOffset + t];
        if (weight == 0.0f) continue;
        const MorphTargetVertex target = scene.MorphTargets(scene.View.MorphDeformSlot)
            [draw.MorphDeformOffset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * float3(target.PositionDelta);
    }
}

// Like ApplyMorphDeform, fetching each target vertex once.
// Authored-morph draws also accumulate the weighted authored normal deltas.
template<typename SetT>
inline void ApplyMorphDeform(const thread SceneT<SetT> &scene, DrawData draw, thread float3 &position, thread float3 &normal_delta, uint vertex_index) {
    if (draw.MorphDeformOffset == INVALID_OFFSET) return;

    const bool authored = draw.MorphShadingAuthored != 0u;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        const float weight = scene.MorphWeights(scene.View.MorphWeightsSlot)[draw.MorphWeightsOffset + t];
        if (weight == 0.0f) continue;
        const MorphTargetVertex target = scene.MorphTargets(scene.View.MorphDeformSlot)
            [draw.MorphDeformOffset + t * draw.VertexCountOrHeadImageSlot + vertex_index];
        position += weight * float3(target.PositionDelta);
        if (authored) normal_delta += weight * float3(target.NormalDelta);
    }
}

#endif
