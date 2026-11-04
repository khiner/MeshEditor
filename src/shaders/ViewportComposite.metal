#pragma once
#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/ViewportCompositePushConstants.h"
#include "tonemapping.metal"

fragment float4 ViewportCompositeFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportThemeUbo &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ViewportCompositePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 overlay = pc.HasOverlay != 0u ? scene.SampleTex(pc.OverlayColorSamplerSlot, in.TexCoord) : float4(0.0f);

    // The view transform acts on unassociated radiance, and premultiplied coverage returns after it.
    const float4 scene_color = scene.SampleTex(pc.SceneColorSamplerSlot, in.TexCoord);
    const float3 radiance = scene_color.a > 0.0f ? scene_color.rgb / scene_color.a : float3(0.0f);
    const float3 scene_display = pc.ViewTransform == 2u ? radiance :
        pc.ViewTransform == 1u ? linearToDisplay(radiance) : linearTosRGB(radiance);
    const float3 base = scene_display * scene_color.a + float3(theme.Colors.Background) * (1.0f - scene_color.a);
    return float4(base * (1.0f - overlay.a) + overlay.rgb, 1.0f);
}
