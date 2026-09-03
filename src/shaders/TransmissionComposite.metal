#ifndef TRANSMISSIONCOMPOSITE_MSL
#define TRANSMISSIONCOMPOSITE_MSL

#include "Bindless.metal"
#include "Varyings.metal"

// Applies exposure to premultiplied opaque-scene radiance before transmission compositing.
fragment float4 TransmissionCompositeFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 prepass = scene.SampleTexLod(view.TransmissionFramebufferSamplerSlot, in.TexCoord, 0.0f);
    return float4(prepass.rgb * view.Exposure, prepass.a);
}

#endif
