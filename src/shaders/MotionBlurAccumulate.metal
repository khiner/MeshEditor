#pragma once
#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/MotionBlurAccumulatePushConstants.h"

fragment float4 MotionBlurAccumulateFragment(
    QuadVaryings in [[stage_in]],
    SCENE_ARGS,
    constant MotionBlurAccumulatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return scene.SampleTex(pc.SceneSamplerSlot, in.TexCoord) * pc.Weight;
}
