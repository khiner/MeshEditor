#ifndef SCENEUBO_MSL
#define SCENEUBO_MSL

#include "Bindless.metal"

constant float PointSize = 8.0f;

inline bool CullBackfaces(constant SceneViewUBO &view, uint double_sided) {
    return view.BackfaceCulling == BackfaceCullingMode::Always ||
        (view.BackfaceCulling == BackfaceCullingMode::Material && double_sided == 0u);
}

inline int BackfaceCullFacing(constant SceneViewUBO &view, uint double_sided, float3 scale) {
    if (!CullBackfaces(view, double_sided)) return 0;
    return scale.x * scale.y * scale.z < 0.0f ? -1 : 1;
}

template<typename SetT>
inline bool IsFrontFacing(const thread SceneT<SetT> &scene, float3 normal, float3 world_pos) {
    return dot(normal, float3(scene.View.CameraPosition) - world_pos) >= 0.0f;
}

// Background rays have no camera translation; orthographic pixels share one direction.
inline float3 WorldBackgroundDirection(const thread Scene &scene, float2 ndc) {
    const float3x3 rotation = scene.View.ViewRotation.Unpack();
    const float4x4 vp = scene.ViewProj();
    const float3x3 projection = float3x3(vp[0].xyz, vp[1].xyz, vp[2].xyz) * transpose(rotation);
    const float3 direction = scene.View.ScreenPixelScale < 0.0f ? float3(0, 0, -1) :
        float3(ndc.x / projection[0][0], ndc.y / projection[1][1], -1.0f);
    return transpose(rotation) * direction;
}

// Precomputed polygon offset factor (matches Blender's GPU_polygon_offset_calc).
// Pushes overlays toward the camera without distance-dependent artifacts.
template<typename SetT>
inline float NdcOffsetFactor(const thread SceneT<SetT> &scene) { return scene.View.NdcOffsetFactor; }

template<typename SetT>
inline float4 WireBaseColor(const thread SceneT<SetT> &scene) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    return float4(float3(scene.View.InteractionMode == InteractionMode::Edit ? colors.WireEdit : colors.Wire), 1.0f);
}

#endif
