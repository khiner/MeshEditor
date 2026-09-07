#ifndef SILHOUETTEEDGECOLOR_MSL
#define SILHOUETTEEDGECOLOR_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "SilhouetteEdgeColorPushConstants.metal"

inline float2 SilhouetteEdge(texture2d<float> silhouette, int2 texel, int width) {
    constexpr sampler pixels(coord::pixel, address::clamp_to_edge, filter::nearest);
    const float2 center = silhouette.read(uint2(texel)).xy;
    float2 nearest{1.0f, 0.0f};
    // Gather four IDs at once; only boundary pixels need their depths.
    const int2 offsets[4] = {int2(0, 1), int2(1, 1), int2(1, 0), int2(0, 0)};
    for (int y = -width; y <= width && (center.y == 0.0f || nearest.y == 0.0f); y += 2) {
        for (int x = -width; x <= width && (center.y == 0.0f || nearest.y == 0.0f); x += 2) {
            const float2 at = float2(texel + int2(x + 1, y + 1));
            const float4 ids = silhouette.gather(pixels, at, int2(0), component::y);
            if (all(ids == center.y)) continue;
            const float4 depths = center.y == 0.0f ? silhouette.gather(pixels, at, int2(0), component::x) : float4(center.x);
            for (uint i = 0; i < 4u; ++i) {
                if (x + offsets[i].x > width || y + offsets[i].y > width || ids[i] == center.y) continue;
                const float2 candidate = center.y != 0.0f ? center : float2(depths[i], ids[i]);
                if (candidate.y != 0.0f && (nearest.y == 0.0f || candidate.x < nearest.x ||
                    (candidate.x == nearest.x && candidate.y < nearest.y))) nearest = candidate;
            }
        }
    }
    return nearest;
}

fragment OverlayTargetsDepth SilhouetteEdgeColorFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SilhouetteEdgeColorPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const int2 texel = int2(in.Position.xy);
    const float2 edge = SilhouetteEdge(bindless.Sampler[pc.SilhouetteSamplerIndex].Texture, texel, int(theme.SilhouetteEdgeWidth));
    const uint object_id = uint(edge.y);
    // Initialize working depth for subsequent overlays without changing visibility depth.
    const float scene_depth = pc.SceneDepthSamplerIndex == INVALID_SLOT ? 1.0f : scene.FetchTex(pc.SceneDepthSamplerIndex, texel, 0).r;
    const float depth = min(scene_depth, edge.x);
    if (object_id == 0u) return {float4(0.0f), depth};

    // UINT32_MAX marks every armature bone instance active because an armature has no single object ID.
    const bool is_active = pc.ActiveObjectId == 0xFFFFFFFFu || (pc.ActiveObjectId != 0u && object_id == pc.ActiveObjectId);
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float4 color = pc.Manipulating != 0u ? float4(float3(colors.Transform), 1.0f) :
        is_active ? float4(float3(colors.ObjectActive), 1.0f) :
                    float4(float3(colors.ObjectSelected), 1.0f);
    return {color, depth};
}

#endif
