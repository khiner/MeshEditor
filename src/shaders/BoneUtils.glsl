#ifndef BONE_UTILS_GLSL
#define BONE_UTILS_GLSL

// Shared utilities for bone overlay shaders.
// Requires: Bindless.glsl (for DrawData, InstanceStateBuffers, SceneViewUBO, ViewportTheme)

uint load_bone_instance_state(DrawData draw) {
    if (draw.InstanceStateSlot != INVALID_SLOT) {
        return uint(InstanceStateBuffers[draw.InstanceStateSlot].States[draw.FirstInstance]);
    }
    return 0u;
}

vec3 bone_wire_color(uint instance_state) {
    bool is_selected = (instance_state & STATE_SELECTED) != 0u;
    bool is_active = (instance_state & STATE_ACTIVE) != 0u;

    if (SceneViewUBO.InteractionMode == InteractionMode_Edit) {
        if (is_active && is_selected) return ViewportTheme.Colors.BoneActive;
        if (is_active) return ViewportTheme.Colors.BoneActiveUnsel;
        if (is_selected) return ViewportTheme.Colors.BoneSelect;
        return ViewportTheme.Colors.WireEdit;
    }
    if (SceneViewUBO.InteractionMode == InteractionMode_Pose) {
        if (is_active && is_selected) return ViewportTheme.Colors.BonePoseActive;
        if (is_active) return ViewportTheme.Colors.BonePoseActiveUnsel;
        if (is_selected) return ViewportTheme.Colors.BonePose;
        return ViewportTheme.Colors.Wire;
    }
    if (is_selected && is_active) return ViewportTheme.Colors.ObjectActive;
    if (is_selected) return ViewportTheme.Colors.ObjectSelected;
    return ViewportTheme.Colors.Wire;
}

// Joint sphere wire color: Vertex/VertexSelected in Edit mode, bone wire color otherwise.
vec3 bone_joint_wire_color(uint instance_state) {
    if (SceneViewUBO.InteractionMode == InteractionMode_Edit) {
        bool is_selected = (instance_state & STATE_SELECTED) != 0u;
        return is_selected ? ViewportTheme.Colors.VertexSelected : ViewportTheme.Colors.Vertex;
    }
    return bone_wire_color(instance_state);
}

struct BoneBillboard {
    vec3 world_pos;
    vec3 center;
    float radius;
};

// Build a camera-facing billboard from a disc mesh vertex and a sphere Transform.
// Disc vertices are expected in XY plane with radius 0.05.
BoneBillboard bone_sphere_billboard(Transform world, vec3 vert_pos) {
    vec3 center = world.P;
    float radius = world.S.x;

    vec3 to_cam = SceneViewUBO.CameraPosition - center;
    float dist = length(to_cam);
    vec3 forward = dist > 0.0 ? to_cam / dist : vec3(0, 0, 1);
    vec3 right = normalize(cross(vec3(0, 1, 0), forward));
    vec3 up = cross(forward, right);

    vec3 world_pos = center + (right * vert_pos.x + up * vert_pos.y) * radius / 0.05;
    return BoneBillboard(world_pos, center, radius);
}

#endif
