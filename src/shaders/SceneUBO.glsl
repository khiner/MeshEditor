#include "SceneViewUBO.glsl"
#include "ViewportTheme.glsl"
#include "InteractionMode.glsl"
#include "Element.glsl"

bool IsFrontFacing(vec3 normal, vec3 world_pos) {
    return dot(normal, SceneViewUBO.CameraPosition - world_pos) >= 0.0;
}
