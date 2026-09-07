#ifndef BONESPHEREWIRE_MSL
#define BONESPHEREWIRE_MSL

// Renders bone-joint outlines around a camera-facing disc.
#include "Bindless.metal"
#include "BoneUtils.metal"
#include "MeshletResolve.metal"
#include "LineQuad.metal"

inline float4 BoneSphereWirePosition(const thread Scene &scene, DrawData draw, uint vertex_id) {
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    const BoneBillboard bb = bone_sphere_billboard(scene, world, float3(vert.Position));
    const float4x4 view_proj = scene.ViewProj();
    float4 clip_pos = view_proj * float4(bb.world_pos, 1.0f);

    // Offset away from the center to avoid overlap with the solid shape (matches Blender).
    const float4 center_clip = view_proj * float4(bb.center, 1.0f);
    const float2 viewport_size = float2(scene.View.ViewportSize);
    const float2 ofs_dir = normalize(clip_pos.xy / clip_pos.w - center_clip.xy / center_clip.w);
    clip_pos.xy += ofs_dir * (1.0f / viewport_size) * clip_pos.w;

    return clip_pos;
}

using BoneSphereWireMeshOutput = metal::mesh<EdgeQuadVaryings, void, 128u, 64u, metal::topology::triangle>;

[[mesh]] void BoneSphereWireMesh(
    BoneSphereWireMeshOutput output,
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
    output.set_primitive_count(64u);
    if (thread_index >= 32u) return;

    DrawData draw = work.Draw;
    draw.IndexSlotOffset = work.Primitive.AuxIndices;
    const auto a = BoneSphereWirePosition(scene, draw, thread_index * 2u);
    const auto b = BoneSphereWirePosition(scene, draw, thread_index * 2u + 1u);
    const float4 color = float4(bone_joint_wire_color(scene, load_bone_instance_state(scene, draw)), 1.0f);
    EmitStroke(output, thread_index, scene, a, b, color);
}

#endif
