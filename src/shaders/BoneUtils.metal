#ifndef BONE_UTILS_MSL
#define BONE_UTILS_MSL

#include "Bindless.metal"

template<typename SetT>
inline uint load_bone_instance_state(const thread SceneT<SetT> &scene, DrawData draw) { return scene.InstanceState(draw); }

template<typename SetT>
inline float3 bone_wire_color(const thread SceneT<SetT> &scene, uint instance_state) {
    const bool is_selected = (instance_state & STATE_SELECTED) != 0u;
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
    constant ViewportThemeColors &colors = scene.Theme.Colors;

    if (scene.View.InteractionMode == InteractionMode_Edit) {
        if (is_active && is_selected) return float3(colors.BoneActive);
        if (is_active) return float3(colors.BoneActiveUnsel);
        if (is_selected) return float3(colors.BoneSelect);
        return float3(colors.WireEdit);
    }
    if (scene.View.InteractionMode == InteractionMode_Pose) {
        if (is_active && is_selected) return float3(colors.BonePoseActive);
        if (is_active) return float3(colors.BonePoseActiveUnsel);
        if (is_selected) return float3(colors.BonePose);
        return float3(colors.Wire);
    }
    if (is_selected && is_active) return float3(colors.ObjectActive);
    if (is_selected) return float3(colors.ObjectSelected);
    return float3(colors.Wire);
}

// Joint sphere wire color: Vertex/VertexSelected in Edit mode, bone wire color otherwise.
template<typename SetT>
inline float3 bone_joint_wire_color(const thread SceneT<SetT> &scene, uint instance_state) {
    if (scene.View.InteractionMode == InteractionMode_Edit) {
        const bool is_selected = (instance_state & STATE_SELECTED) != 0u;
        return is_selected ? float3(scene.Theme.Colors.VertexSelected) : float3(scene.Theme.Colors.Vertex);
    }
    return bone_wire_color(scene, instance_state);
}

struct BoneBillboard {
    float3 world_pos;
    float3 center;
    float radius;
};

// Returns a camera-facing billboard position for an XY-plane disc with radius 0.05.
template<typename SetT>
inline BoneBillboard bone_sphere_billboard(const thread SceneT<SetT> &scene, Transform world, float3 vert_pos) {
    const float3 center = float3(world.P);
    const float radius = world.S[0];

    const float3 to_cam = float3(scene.View.CameraPosition) - center;
    const float dist = length(to_cam);
    const float3 forward = dist > 0.0f ? to_cam / dist : float3(0, 0, 1);
    const float3 right = normalize(cross(float3(0, 1, 0), forward));
    const float3 up = cross(forward, right);

    const float3 world_pos = center + (right * vert_pos.x + up * vert_pos.y) * radius / 0.05f;
    return BoneBillboard{world_pos, center, radius};
}

#endif
