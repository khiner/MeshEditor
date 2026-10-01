#ifndef MORPH_DEFORM_MSL
#define MORPH_DEFORM_MSL

// Applies morph-target position deformation in mesh-local space and returns the input when no morph data exists.
#include "Bindless.metal"

// `weights_slot` selects current, shutter-open, or shutter-close pose weights.
template<typename SetT>
inline void ApplyMorphDeform(const thread SceneT<SetT> &scene, DrawData draw, thread float3 &position, uint vertex_index, uint weights_slot) {
    if (draw.MorphDeformOffset == InvalidOffset) return;

    const uint handle=draw.VertexOffset+vertex_index;

    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        const float weight = scene.MorphWeights(weights_slot)[draw.MorphWeightsOffset + t];
        if (weight == 0.0f) continue;
        const MorphTargetVertex target=scene.MorphTargets(scene.View.Morph.ValuesSlot)[ElementAttributeIndex(BindlessBuffer(uint,scene.B.Buffer,scene.View.Morph.BlocksSlot),handle,t)];
        position += weight * float3(target.PositionDelta);
    }
}

// Applies position deformation and accumulates weighted authored normal deltas with one fetch per target vertex.
template<typename SetT>
inline void ApplyMorphDeform(const thread SceneT<SetT> &scene, DrawData draw, thread float3 &position, thread float3 &normal_delta, uint vertex_index) {
    if (draw.MorphDeformOffset == InvalidOffset) return;

    const uint handle=draw.VertexOffset+vertex_index;

    const bool authored = draw.MorphShadingAuthored != 0u;
    for (uint t = 0; t < draw.MorphTargetCount; ++t) {
        const float weight = scene.MorphWeights(scene.View.MorphWeightsSlot)[draw.MorphWeightsOffset + t];
        if (weight == 0.0f) continue;
        const MorphTargetVertex target=scene.MorphTargets(scene.View.Morph.ValuesSlot)[ElementAttributeIndex(BindlessBuffer(uint,scene.B.Buffer,scene.View.Morph.BlocksSlot),handle,t)];
        position += weight * float3(target.PositionDelta);
        if (authored) normal_delta += weight * float3(target.NormalDelta);
    }
}

#endif
