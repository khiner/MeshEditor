#ifndef SILHOUETTEEDGEDEPTHOBJECT_MSL
#define SILHOUETTEEDGEDEPTHOBJECT_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "SilhouetteEdgeDepthObjectPushConstants.metal"

struct SilhouetteEdgeTarget {
    float ObjectId [[color(0)]];
    float Depth [[depth(any)]];
};

fragment SilhouetteEdgeTarget SilhouetteEdgeDepthObjectFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SilhouetteEdgeDepthObjectPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const int2 tex_size = int2(scene.TexSize(pc.SilhouetteSamplerIndex, 0));
    const int2 texel = int2(in.TexCoord * float2(tex_size));
    const float2 depth_id = scene.FetchTex(pc.SilhouetteSamplerIndex, texel, 0).xy;

    float2 min_depth_id = float2(10, 0);
    const int edge_width = int(scene.Theme.SilhouetteEdgeWidth);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            // Sparse neighborhood sampling approximates small outline widths; the sampler clamps boundary reads.
            const int2 neighbor_texel = texel + int2(i, j) * edge_width;
            const float2 neighbor_depth_id = scene.FetchTex(pc.SilhouetteSamplerIndex, neighbor_texel, 0).xy;
            if (depth_id.y != neighbor_depth_id.y) {
                const float2 edge_depth_id = depth_id.x != 0.0f ? depth_id : neighbor_depth_id;
                min_depth_id = float2(min(min_depth_id.x, edge_depth_id.x), edge_depth_id.y);
            }
        }
    }
    if (min_depth_id.y == 0.0f) discard_fragment();

    return SilhouetteEdgeTarget{min_depth_id.y, min_depth_id.x};
}

#endif
