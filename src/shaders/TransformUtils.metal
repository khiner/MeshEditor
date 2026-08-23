#ifndef TRANSFORMUTILS_MSL
#define TRANSFORMUTILS_MSL

// Shared pending-transform helpers.
#include "Bindless.metal"

inline float4 quat_conjugate(float4 q) { return float4(-q.xyz, q.w); }

template<typename SetT>
inline float3 apply_pending_transform_world(const thread SceneT<SetT> &scene, float3 world_pos) {
    float3 offset = world_pos - float3(scene.View.PendingPivot);
    offset = float3(scene.View.PendingScale) * offset;
    offset = quat_rotate(float4(scene.View.PendingRotation), offset);
    return float3(scene.View.PendingPivot) + offset + float3(scene.View.PendingTranslation);
}

inline float3 trs_inverse_transform_point(Transform t, float3 pos) {
    return quat_rotate(quat_conjugate(float4(t.R)), pos - float3(t.P)) / float3(t.S);
}

// Object-mode gizmo preview: selected instances' world positions follow the pending transform.
// The pose pre-pass bakes edit-mode vertex previews into posed positions.
template<typename SetT>
inline float3 apply_object_pending_transform(const thread SceneT<SetT> &scene, DrawData draw, float3 world_pos) {
    if (scene.View.IsTransforming == 0u || scene.View.InteractionMode == InteractionMode_Edit || draw.InstanceStateSlot == INVALID_SLOT) return world_pos;
    const uint instance_state = uint(scene.InstanceStates(draw.InstanceStateSlot)[draw.FirstInstance]);
    if ((instance_state & STATE_SELECTED) == 0u) return world_pos;
    return apply_pending_transform_world(scene, world_pos);
}

// Clip position of the draw's vertex: local position through the world transform, any pending
// object transform, and the view projection.
template<typename SetT>
inline float4 MeshletPosition(const thread SceneT<SetT> &scene, DrawData draw, Transform world, uint vertex_id) {
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, vertex_id)));
    return scene.ViewProj() * float4(world_pos, 1.0f);
}

#endif
