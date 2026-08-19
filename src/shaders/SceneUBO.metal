#ifndef SCENEUBO_MSL
#define SCENEUBO_MSL

#include "Bindless.metal"

// Diameter in pixels of a drawn point.
constant float PointSize = 8.0f;

template<typename SetT>
inline bool IsFrontFacing(const thread SceneT<SetT> &scene, float3 normal, float3 world_pos) {
    return dot(normal, float3(scene.View.CameraPosition) - world_pos) >= 0.0f;
}

// Precomputed polygon offset factor (matches Blender's GPU_polygon_offset_calc).
// Pushes overlays toward the camera without distance-dependent artifacts.
template<typename SetT>
inline float NdcOffsetFactor(const thread SceneT<SetT> &scene) { return scene.View.NdcOffsetFactor; }

#endif
