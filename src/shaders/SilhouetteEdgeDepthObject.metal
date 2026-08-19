#ifndef SILHOUETTEEDGEDEPTHOBJECT_MSL
#define SILHOUETTEEDGEDEPTHOBJECT_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "SilhouetteEdgeDepthObjectPushConstants.metal"

// The object id for edge pixels, with the depth that places the outline.
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
    // The source holds {Depth, ObjectID} at each pixel.
    const int2 tex_size = int2(scene.TexSize(pc.SilhouetteSamplerIndex, 0));
    const int2 texel = int2(in.TexCoord * float2(tex_size));
    const float2 depth_id = scene.FetchTex(pc.SilhouetteSamplerIndex, texel, 0).xy;

    float2 min_depth_id = float2(10, 0); // The nearest depth over the neighbourhood.
    const int edge_width = int(scene.Theme.SilhouetteEdgeWidth);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            // Jumping over pixels in the neighbourhood test extends the edge width without looping
            // over every pixel in it. That looks decent at small widths, and beyond those the
            // missing corner pixels start to show.
            // The sampler clamps to edge, so out-of-bounds reads need no guard.
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
