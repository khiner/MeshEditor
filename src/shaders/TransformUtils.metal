#ifndef TRANSFORMUTILS_MSL
#define TRANSFORMUTILS_MSL

// Shared pending-transform and instance-bounds helpers.
#include "AABB.metal"
#include "Bindless.metal"
#include "Frustum.metal"

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

// An instance's local AABB carried into world space as an oriented box.
struct OrientedBounds {
    float3 Center;
    float3 Ax, Ay, Az;
    bool Valid;
};

inline OrientedBounds TransformBounds(AABB bounds, Transform world) {
    const float3 lo = float3(bounds.Min), hi = float3(bounds.Max);
    if (lo.x > hi.x) return {};
    const float3 half_local = (hi - lo) * 0.5f;
    const float3 scale = float3(world.S);
    const float4 rotation = float4(world.R);
    return {
        trs_transform_point(world, (lo + hi) * 0.5f),
        quat_rotate(rotation, float3(scale.x * half_local.x, 0, 0)),
        quat_rotate(rotation, float3(0, scale.y * half_local.y, 0)),
        quat_rotate(rotation, float3(0, 0, scale.z * half_local.z)),
        true,
    };
}

// True unless the draw's instance bounds lie wholly outside the view frustum.
// A pending transform previews instances beyond their recorded bounds, so it suspends the test.
template<typename SetT>
inline bool InstanceInFrustum(const thread SceneT<SetT> &scene, DrawData draw) {
    if (scene.View.InstanceBoundsSlot == INVALID_SLOT || scene.View.IsTransforming != 0u) return true;
    const auto bounds = TransformBounds(
        BindlessBuffer(AABB, scene.B.Buffer, scene.View.InstanceBoundsSlot)[draw.FirstInstance],
        scene.Models(draw.ModelSlot)[draw.FirstInstance]
    );
    return !bounds.Valid || in_frustum(scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az);
}

#endif
