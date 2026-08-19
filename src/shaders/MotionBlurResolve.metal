#ifndef MOTIONBLURRESOLVE_MSL
#define MOTIONBLURRESOLVE_MSL

#include "Bindless.metal"
#include "Varyings.metal"

struct MotionBlurResolvePushConstants {
    uint AccumSamplerSlot;
    float InvSteps; // 1 / step count.
};

// Average the summed steps. Color and coverage are both premultiplied, so both scale together.
fragment float4 MotionBlurResolveFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurResolvePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return scene.SampleTex(pc.AccumSamplerSlot, in.TexCoord) * pc.InvSteps;
}

#endif
