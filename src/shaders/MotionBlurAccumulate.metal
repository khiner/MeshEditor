#ifndef MOTIONBLURACCUMULATE_MSL
#define MOTIONBLURACCUMULATE_MSL

#include "Bindless.metal"
#include "Varyings.metal"

struct MotionBlurAccumulatePushConstants {
    uint GatherSamplerSlot;
};

// Sum this step's blurred scene into the accumulation target through the additive blend. The gather
// output is premultiplied, so alpha sums alongside color and a blurred edge keeps its partial
// coverage of the backdrop.
fragment float4 MotionBlurAccumulateFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurAccumulatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return scene.SampleTex(pc.GatherSamplerSlot, in.TexCoord);
}

#endif
