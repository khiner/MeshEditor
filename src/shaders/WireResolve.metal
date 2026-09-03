#ifndef WIRERESOLVE_MSL
#define WIRERESOLVE_MSL

// Resolves per-class coverage to premultiplied overlay color and nearest wire depth.
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

fragment OverlayTargetsDepth WireResolveFragment(
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
    const uint base = (pixel.y * extent.x + pixel.x) * WireCoverage_WordsPerPixel;

    // Select color from the highest-coverage class and alpha from total coverage.
    float total = 0.0f;
    float4 color = float4(0.0f);
    float best = 0.0f;
    for (uint wire_class = 0u; wire_class < WireCoverage_DepthWord; ++wire_class) {
        const float coverage = float(words[base + wire_class]) * WireResolveScale;
        if (coverage <= 0.0f) continue;
        total += coverage;
        if (coverage >= best) {
            best = coverage;
            color = WireClassColor(scene, wire_class);
        }
    }
    if (total <= 0.0f) discard_fragment();

    const float alpha = saturate(total) * color.a;
    OverlayTargetsDepth out;
    out.Color = float4(color.rgb * alpha, alpha);
    out.LineData = float4(0.0f);
    out.Depth = as_type<float>(~words[base + WireCoverage_DepthWord]);
    return out;
}

#endif
