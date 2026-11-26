#pragma once
#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/MotionBlurResolvePushConstants.h"

// Averages the summed steps, scaling premultiplied color and coverage together.
fragment float4 MotionBlurResolveFragment(
    QuadVaryings in [[stage_in]],
    SCENE_ARGS,
    constant MotionBlurResolvePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return scene.SampleTex(pc.AccumSamplerSlot, in.TexCoord) * pc.InvSteps;
}
