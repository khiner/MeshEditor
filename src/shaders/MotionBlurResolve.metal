#pragma once
#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/MotionBlurResolvePushConstants.h"

// Averages the summed steps, scaling premultiplied color and coverage together.
fragment float4 MotionBlurResolveFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurResolvePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return scene.SampleTex(pc.AccumSamplerSlot, in.TexCoord) * pc.InvSteps;
}
