#ifndef PYRAMIDOCCLUSION_MSL
#define PYRAMIDOCCLUSION_MSL

#include "Bindless.metal"
#include "ScreenSpace.metal"

// The projected diameter in pixels of a world-space sphere.
inline float ProjectedDiameterPixels(const thread Scene &scene, float3 center, float radius) {
    if (scene.View.ScreenPixelScale <= 0.0f) return 2.0f * radius / -scene.View.ScreenPixelScale;
    const float distance = max(length(center - float3(scene.View.CameraPosition)) - radius, scene.View.CameraNear);
    return 2.0f * radius / (distance * scene.View.ScreenPixelScale);
}

// Whether the depth pyramid hides a world-space box whose raster footprint grows by `margin` pixels.
// A positive `depth_pull` tests the box at its nearest depth after that clip-space pull toward the camera.
inline bool BoxOccluded(
    const thread Scene &scene, uint pyramid_slot, float3 center, float3 ax, float3 ay, float3 az, float margin, float depth_pull
) {
    const float4x4 view_proj = scene.ViewProj();
    float2 uv_min = float2(1e30f), uv_max = float2(-1e30f);
    float min_depth = 1e30f;
    for (uint c = 0; c < 8; ++c) {
        const float3 corner = center + ((c & 1u) ? ax : -ax) + ((c & 2u) ? ay : -ay) + ((c & 4u) ? az : -az);
        const float4 clip = view_proj * float4(corner, 1.0f);
        if (clip.w <= 0.0f) return false;
        const float3 ndc = clip.xyz / clip.w;
        const float2 uv = ndc_to_uv(ndc.xy);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        min_depth = min(min_depth, depth_pull > 0.0f ? (clip.z - depth_pull) / clip.w - 5e-7f : ndc.z);
    }
    if (min_depth <= 0.0f) return false;
    const float2 viewport_size = float2(scene.View.ViewportSize);
    // The pyramid's first level is half resolution.
    const float2 min_px = clamp(uv_min * viewport_size - margin, 0.0f, viewport_size) * 0.5f;
    const float2 max_px = clamp(uv_max * viewport_size + margin, 0.0f, viewport_size) * 0.5f;
    const float max_dim = max(max_px.x - min_px.x, max_px.y - min_px.y);
    const int mip_count = int(scene.B.Sampler[pyramid_slot].Texture.get_num_mip_levels());
    const int level = clamp(int(ceil(log2(max(max_dim * 0.5f, 1.0f)))), 0, mip_count - 1);
    const int2 data_max = (int2(viewport_size) - 1) >> (level + 1);
    const int2 lo = clamp(int2(min_px) >> level, int2(0), data_max);
    const int2 hi = clamp(int2(max_px) >> level, int2(0), data_max);
    if (hi.x - lo.x > 3 || hi.y - lo.y > 3) return false;
    float occluder = 0.0f;
    for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) occluder = max(occluder, scene.FetchTex(pyramid_slot, int2(x, y), uint(level)).r);
    }
    return min_depth > occluder;
}

#endif
