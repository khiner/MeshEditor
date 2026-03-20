#include "SceneViewUBO.glsl"
#include "ViewportTheme.glsl"
#include "InteractionMode.glsl"
#include "Element.glsl"

bool IsFrontFacing(vec3 normal, vec3 world_pos) {
    return dot(normal, SceneViewUBO.CameraPosition - world_pos) >= 0.0;
}

// Precomputed polygon offset factor (matches Blender's GPU_polygon_offset_calc).
// Used to push overlays toward the camera without distance-dependent artifacts.
float NdcOffsetFactor() {
    return SceneViewUBO.NdcOffsetFactor;
}
