#ifndef BONEWIRE_MSL
#define BONEWIRE_MSL

// Port of Blender's overlay_armature_shape_outline_vert.glsl.
// Each edge loads 4 adjacency indices: [adj_left, edge_v0, edge_v1, adj_right].
#include "Bindless.metal"
#include "BoneUtils.metal"
#include "MeshletResolve.metal"
#include "Varyings.metal"

inline LineVaryings DiscardedEdge() {
    return LineVaryings{float4(0, 0, -2, 1), float4(0), float2(0), float2(0)};
}

inline LineVaryings BoneWireMeshVertexAt(const thread Scene &scene, DrawData draw, uint vertex_id) {
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    const float4x4 M = trs_to_mat4(world);

    const float3x3 VR = scene.View.ViewRotation.Unpack();
    const float4x4 MV = float4x4(
        float4(VR[0], 0), float4(VR[1], 0), float4(VR[2], 0),
        float4(-(VR * float3(scene.View.CameraPosition)), 1)
    ) * M;

    const uint edge_index = vertex_id / 2u;
    const uint vert_in_edge = vertex_id % 2u;

    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const uint adj_base = draw.IndexSlotOffset.Offset + edge_index * 4u;
    const uint i0 = indices[adj_base + 0u];
    const uint i1 = indices[adj_base + 1u];
    const uint i2 = indices[adj_base + 2u];
    const uint i3 = indices[adj_base + 3u];

    device const Vertex *vertices = scene.Vertices(draw.VertexSlot);
    const float3 p0 = float3(vertices[i0 + draw.VertexOffset].Position);
    const float3 p1 = float3(vertices[i1 + draw.VertexOffset].Position);
    const float3 p2 = float3(vertices[i2 + draw.VertexOffset].Position);
    const float3 p3 = float3(vertices[i3 + draw.VertexOffset].Position);

    const float3 vs0 = (MV * float4(p0, 1.0f)).xyz;
    const float3 vs1 = (MV * float4(p1, 1.0f)).xyz;
    const float3 vs2 = (MV * float4(p2, 1.0f)).xyz;
    const float3 vs3 = (MV * float4(p3, 1.0f)).xyz;

    const float3 view_vec = normalize(vs1);

    const float3 v10 = vs0 - vs1;
    const float3 v12 = vs2 - vs1;
    const float3 v13 = vs3 - vs1;

    // cross(v10, v12) points outward under the face winding convention.
    float3 n0 = cross(v10, v12);
    const float len_n0 = length(n0);
    n0 = len_n0 > 0.0f ? n0 / len_n0 : float3(0);
    float3 n3 = cross(v12, v13);
    const float len_n3 = length(n3);
    n3 = len_n3 > 0.0f ? n3 / len_n3 : float3(0);

    const float fac0 = dot(view_vec, n0);
    const float fac3 = dot(view_vec, n3);

    // Treat a face perpendicular to the view as an outline boundary.
    if (abs(fac0) > 1e-5f && abs(fac3) > 1e-5f) {
        if (sign(fac0) == sign(fac3)) return DiscardedEdge();
    }

    // Omit concave edges from the outline.
    const bool inverted = dot(cross(M[0].xyz, M[1].xyz), M[2].xyz) < 0.0f;
    const float3 n0_check = inverted ? -n0 : n0;
    if (dot(n0_check, v13) > 0.0001f) return DiscardedEdge();

    // Match BoneSolid's projection sequence to preserve bit-identical depth.
    const float3 world_pos = (M * float4(vert_in_edge == 0u ? p1 : p2, 1.0f)).xyz;
    float4 clip_pos = scene.ViewProj() * float4(world_pos, 1.0f);

    LineVaryings out;
    out.Color = float4(bone_wire_color(scene, load_bone_instance_state(scene, draw)), 1.0f);

    // Apply Blender's depth bias to prevent z-fighting with the fill.
    clip_pos.z -= 1e-4f;

    out.Position = clip_pos;
    const float2 screen_pos = clip_to_frag_co(clip_pos, float2(scene.View.ViewportSize));
    out.EdgeStart = screen_pos;
    out.EdgePos = screen_pos;
    return out;
}

using BoneWireMeshOutput = metal::mesh<LineVaryings, void, 24u, 12u, metal::topology::line>;

[[mesh]] void BoneWireMesh(
    BoneWireMeshOutput output,
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
    output.set_primitive_count(12u);
    if (thread_index >= 24u) return;

    DrawData draw = work.Draw;
    draw.IndexSlotOffset = work.Primitive.AuxIndices;
    output.set_vertex(thread_index, BoneWireMeshVertexAt(scene, draw, thread_index));
    output.set_index(thread_index, thread_index);
}


#endif
