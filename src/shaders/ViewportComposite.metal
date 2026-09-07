#ifndef VIEWPORTCOMPOSITE_MSL
#define VIEWPORTCOMPOSITE_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "tonemapping.metal"

struct ViewportCompositePushConstants {
    uint SceneColorSamplerSlot;
    uint OverlayColorSamplerSlot;
    // View transform: 0 encodes, 1 tone maps and encodes, 2 preserves debug values.
    uint ViewTransform;
    uint HasOverlay;
    packed_float4 Backdrop;
};

fragment float4 ViewportCompositeFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ViewportCompositePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 overlay = pc.HasOverlay != 0u ? scene.SampleTex(pc.OverlayColorSamplerSlot, in.TexCoord) : float4(0.0f);

    // Apply the view transform to unassociated radiance, then restore premultiplied coverage.
    const float4 scene_color = scene.SampleTex(pc.SceneColorSamplerSlot, in.TexCoord);
    const float3 radiance = scene_color.a > 0.0f ? scene_color.rgb / scene_color.a : float3(0.0f);
    const float3 scene_display = pc.ViewTransform == 2u ? radiance :
        pc.ViewTransform == 1u ? linearToDisplay(radiance) : linearTosRGB(radiance);
    // Composite the display-space UI backdrop after the scene view transform.
    const float3 base = scene_display * scene_color.a + float4(pc.Backdrop).rgb * (1.0f - scene_color.a);
    return float4(base * (1.0f - overlay.a) + overlay.rgb, 1.0f);
}

#endif
