#ifndef EDGEQUAD_MSL
#define EDGEQUAD_MSL

#include "Bindless.metal"
#include "EditOverlayConstant.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

// Matches Blender's overlay_shader_shared.hh constant values.
constant float EdgeQuadDiscRadius = 0.5641895835477563f * 1.05f;
inline float EdgeQuadSmoothWeight(float distance) {
    return smoothstep(0.5f - EdgeQuadDiscRadius, 0.5f + EdgeQuadDiscRadius, distance);
}

inline OverlayTargets ShadeEdgeQuad(EdgeQuadVaryings in, const thread Scene &scene, bool include_outer) {
    const float edge_width = scene.Theme.EdgeWidth;
    const float dist = abs(in.EdgeCoord) - max(edge_width - 0.5f, 0.0f);
    const float mix_w = EdgeQuadSmoothWeight(dist);
    float4 color = in.Color;
    if (include_outer && in.OuterColor.a > 0.0f) {
        color = mix(in.OuterColor, color, 1.0f - mix_w);
        color.a *= 1.0f - EdgeQuadSmoothWeight(dist - max(edge_width, 1.0f));
    } else {
        color.a *= 1.0f - mix_w;
    }
    // Edge quads apply antialiasing before the composite pass.
    return OverlayTargets{color, float4(0.0f)};
}

fragment OverlayTargets EdgeQuadFragment(
    EdgeQuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return ShadeEdgeQuad(in, scene, IncludeOuter);
}

#endif
