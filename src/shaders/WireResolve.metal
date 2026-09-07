#ifndef WIRERESOLVE_MSL
#define WIRERESOLVE_MSL

// Composite base, incidental, selected, and active coverage in that display order.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "WireCoverage.metal"
#include "WireResolvePushConstants.metal"

constant float WireResolveScale = 1.0f / 255.0f;

inline float4 WireClassColor(const thread Scene &scene, uint wire_class) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        if (wire_class == WireCoverage_Active) return float4(float3(colors.ObjectActive), 1.0f);
        if (wire_class == WireCoverage_Selected) return float4(float3(colors.ObjectSelected), 1.0f);
        return WireBaseColor(scene);
    }
    if (wire_class == WireCoverage_Active) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if (wire_class == WireCoverage_Selected) return float4(float3(colors.EdgeSelected), 1.0f);
    if (wire_class == WireCoverage_Incidental) return float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    return WireBaseColor(scene);
}

fragment float4 WireResolveFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant WireResolvePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint2 extent = uint2(scene.View.ViewportSize);
    const uint2 pixel = uint2(in.Position.xy);
    device const uint *words = BindlessBuffer(uint, bindless.Buffer, pc.CoverageSlot);
    const uint coverage = words[pixel.y * extent.x + pixel.x];
    if (coverage == 0u) discard_fragment();
    float4 color = float4(0.0f);
    for (uint wire_class = 0u; wire_class < 4u; ++wire_class) {
        const float4 layer = WireClassColor(scene, wire_class);
        const float alpha = float((coverage >> (wire_class * 8u)) & 255u) * WireResolveScale * layer.a;
        color = float4(layer.rgb * alpha, alpha) + (1.0f - alpha) * color;
    }
    return color;
}

#endif
