#pragma once
#include "Bindless.metal"
#include "Varyings.metal"

// Applies exposure to premultiplied opaque-scene radiance before transmission compositing.
fragment float4 TransmissionCompositeFragment(
    QuadVaryings in [[stage_in]],
    SCENE_ARGS
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 prepass = scene.SampleTexLod(view.TransmissionFramebufferSamplerSlot, in.TexCoord, 0.0f);
    return float4(prepass.rgb * view.Exposure, prepass.a);
}
