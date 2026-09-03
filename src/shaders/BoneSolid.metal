#ifndef BONESOLID_MSL
#define BONESOLID_MSL

#include "Bindless.metal"
#include "BoneUtils.metal"
#include "MeshletResolve.metal"
#include "Varyings.metal"

inline BoneSolidVaryings BoneSolidMeshVertexAt(const thread Scene &scene, DrawData draw, uint vertex_id) {
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    const float4x4 M = trs_to_mat4(world);
    const float3 world_pos = (M * float4(float3(vert.Position), 1.0f)).xyz;
    const float3x3 M3 = float3x3(M[0].xyz, M[1].xyz, M[2].xyz);
    const float3 view_normal = scene.View.ViewRotation.Unpack() * normalize(M3 * scene.GetVertexNormal(draw, idx));

    const bool is_object_mode = scene.View.InteractionMode == InteractionMode_Object;
    const float3 bone_color = float3(scene.Theme.Colors.BoneSolid);
    const float3 hint_color = is_object_mode ? bone_color : bone_wire_color(scene, load_bone_instance_state(scene, draw));
    const float3 state_color = hint_color * hint_color * 0.1f; // Matches Blender's bone_hint_color_shade.

    const float3 light = normalize(float3(0.1f, 0.1f, 0.8f));
    const float fac = clamp(dot(view_normal, light) * 0.8f + 0.2f, 0.0f, 1.0f);
    const float alpha = scene.View.BoneXRay != 0u ? 0.6f : 1.0f;

    BoneSolidVaryings out;
    out.Color = float4(mix(state_color, bone_color, fac * fac), alpha);
    out.Inverted = int(dot(cross(M[0].xyz, M[1].xyz), M[2].xyz) < 0.0f);
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    return out;
}

using BoneSolidMeshOutput = metal::mesh<BoneSolidVaryings, void, 24u, 8u, metal::topology::triangle>;

[[mesh]] void BoneSolidMesh(
    BoneSolidMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc, threadgroup_position.x);
    if (!work.Valid) {
        if (thread_index == 0u) output.set_primitive_count(0u);
        return;
    }
    output.set_primitive_count(8u);
    if (thread_index >= 24u) return;

    output.set_vertex(thread_index, BoneSolidMeshVertexAt(scene, work.Draw, thread_index));
    output.set_index(thread_index, thread_index);
}


fragment OverlayTargetsDepth BoneSolidFragment(
    BoneSolidVaryings in [[stage_in]],
    bool front_facing [[front_facing]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    // Account for mirrored instances during manual backface culling.
    if ((in.Inverted == 1) == front_facing) discard_fragment();

    OverlayTargetsDepth out;
    out.Color = in.Color;
    out.LineData = float4(0);
    // X-ray depth passes the cleared 1.0 depth test without occluding wires.
    out.Depth = view.BoneXRay != 0u ? 0.999999f : in.Position.z;
    return out;
}

#endif
