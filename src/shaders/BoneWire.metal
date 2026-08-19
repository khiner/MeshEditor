#ifndef BONEWIRE_MSL
#define BONEWIRE_MSL

// Per-bone silhouette outline using adjacency-based edge detection.
// Port of Blender's overlay_armature_shape_outline_vert.glsl.
// Drawn as a line list: each output pair of vertices is one edge.
// vertex_id / 2 is the edge index, vertex_id % 2 is the endpoint.
// Each edge loads 4 adjacency indices: [adj_left, edge_v0, edge_v1, adj_right].
#include "Bindless.metal"
#include "BoneUtils.metal"
#include "Varyings.metal"
#include "MainDrawPushConstants.metal"

// A degenerate position behind the near plane, for edges the silhouette test rejects.
inline LineVaryings DiscardedEdge() {
    return LineVaryings{float4(0, 0, -2, 1), float4(0), float2(0), float2(0)};
}

vertex LineVaryings BoneWireVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MainDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    const float4x4 M = trs_to_mat4(world);

    const float3x3 VR = view.ViewRotation.Unpack();
    const float4x4 MV = float4x4(
        float4(VR[0], 0), float4(VR[1], 0), float4(VR[2], 0),
        float4(-(VR * float3(view.CameraPosition)), 1)
    ) * M;

    const uint edge_index = vertex_id / 2u;
    const uint vert_in_edge = vertex_id % 2u;

    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const uint adj_base = draw.IndexSlotOffset.Offset + edge_index * 4u;
    const uint i0 = indices[adj_base + 0u]; // adj left
    const uint i1 = indices[adj_base + 1u]; // edge v0
    const uint i2 = indices[adj_base + 2u]; // edge v1
    const uint i3 = indices[adj_base + 3u]; // adj right

    device const Vertex *vertices = scene.Vertices(draw.VertexSlot);
    const float3 p0 = float3(vertices[i0 + draw.VertexOffset].Position);
    const float3 p1 = float3(vertices[i1 + draw.VertexOffset].Position);
    const float3 p2 = float3(vertices[i2 + draw.VertexOffset].Position);
    const float3 p3 = float3(vertices[i3 + draw.VertexOffset].Position);

    // View space, following Blender's view-space face normals.
    const float3 vs0 = (MV * float4(p0, 1.0f)).xyz;
    const float3 vs1 = (MV * float4(p1, 1.0f)).xyz;
    const float3 vs2 = (MV * float4(p2, 1.0f)).xyz;
    const float3 vs3 = (MV * float4(p3, 1.0f)).xyz;

    // View vector: perspective points toward edge_v0.
    const float3 view_vec = normalize(vs1);

    // Edge vectors from edge_v0.
    const float3 v10 = vs0 - vs1; // toward adj_left
    const float3 v12 = vs2 - vs1; // toward edge_v1
    const float3 v13 = vs3 - vs1; // toward adj_right

    // Face normals of the two triangles sharing this edge.
    // cross(v10, v12) points outward, matching the face winding convention.
    float3 n0 = cross(v10, v12); // left face
    const float len_n0 = length(n0);
    n0 = len_n0 > 0.0f ? n0 / len_n0 : float3(0);
    float3 n3 = cross(v12, v13); // right face
    const float len_n3 = length(n3);
    n3 = len_n3 > 0.0f ? n3 / len_n3 : float3(0);

    const float fac0 = dot(view_vec, n0);
    const float fac3 = dot(view_vec, n3);

    // A face perpendicular to the view counts as an outline edge.
    // Otherwise two faces turned the same way make the edge internal.
    if (abs(fac0) > 1e-5f && abs(fac3) > 1e-5f) {
        if (sign(fac0) == sign(fac3)) return DiscardedEdge();
    }

    // Concavity check: concave edges are not outlined.
    const bool inverted = dot(cross(M[0].xyz, M[1].xyz), M[2].xyz) < 0.0f;
    const float3 n0_check = inverted ? -n0 : n0;
    if (dot(n0_check, v13) > 0.0001f) return DiscardedEdge();

    // The chosen endpoint, with depth computed as in BoneSolid (extract xyz, repack with w=1)
    // so the two agree bit for bit.
    const float3 world_pos = (M * float4(vert_in_edge == 0u ? p1 : p2, 1.0f)).xyz;
    float4 clip_pos = scene.ViewProj() * float4(world_pos, 1.0f);

    LineVaryings out;
    out.Color = float4(bone_wire_color(scene, load_bone_instance_state(scene, draw)), 1.0f);

    // Depth bias: push the wire in front of the fill to avoid z-fighting (matches Blender).
    clip_pos.z -= 1e-4f;

    out.Position = clip_pos;
    const float2 screen_pos = clip_to_frag_co(clip_pos, float2(view.ViewportSize));
    out.EdgeStart = screen_pos;
    out.EdgePos = screen_pos;
    return out;
}

#endif
