#ifndef OVERLAYFADE_MSL
#define OVERLAYFADE_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"

// Scales an overlay fragment behind the scene surface by the view's X-ray fade.
// Zero means the depth test already rejected occluded fragments.
inline float OverlayBehindFade(const thread Scene &scene, float4 position) {
    const float opacity = scene.View.OverlayBehindOpacity;
    if (opacity <= 0.0f || opacity >= 1.0f) return 1.0f;
    const float scene_depth = scene.FetchTex(scene.View.SceneDepthSamplerSlot, int2(position.xy), 0u).r;
    return position.z > scene_depth ? opacity : 1.0f;
}

#endif
