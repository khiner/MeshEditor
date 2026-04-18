#version 450

// Per-bone silhouette outline using adjacency-based edge detection.
// Port of Blender's overlay_armature_shape_outline_vert.glsl.
// Drawn as eLineList; each output pair of vertices = one edge.
// gl_VertexIndex / 2 = edge index, gl_VertexIndex % 2 = which endpoint.
// Each edge loads 4 adjacency indices: [adj_left, edge_v0, edge_v1, adj_right].

#include "Bindless.glsl"
#include "BoneUtils.glsl"

layout(location = 2) out vec4 Color;
layout(location = 12) flat out vec2 EdgeStart;
layout(location = 13) out vec2 EdgePos;

void main() {
    const DrawData draw = GetDrawData();
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];
    const mat4 M = trs_to_mat4(world);

    // Build view matrix from SceneViewUBO
    const mat3 VR = SceneViewUBO.ViewRotation;
    const mat4 MV = mat4(
        vec4(VR[0], 0), vec4(VR[1], 0), vec4(VR[2], 0),
        vec4(-VR * SceneViewUBO.CameraPosition, 1)
    ) * M;

    // Each edge uses 4 adjacency indices; gl_VertexIndex cycles through pairs
    const uint edge_index = uint(gl_VertexIndex) / 2u;
    const uint vert_in_edge = uint(gl_VertexIndex) % 2u;

    // Load 4 positions for this edge from the adjacency index buffer
    const uint adj_base = draw.IndexSlotOffset.Offset + edge_index * 4u;
    const uint i0 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[adj_base + 0u]; // adj left
    const uint i1 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[adj_base + 1u]; // edge v0
    const uint i2 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[adj_base + 2u]; // edge v1
    const uint i3 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[adj_base + 3u]; // adj right

    const vec3 p0 = VertexBuffers[draw.VertexSlot].Vertices[i0 + draw.VertexOffset].Position;
    const vec3 p1 = VertexBuffers[draw.VertexSlot].Vertices[i1 + draw.VertexOffset].Position;
    const vec3 p2 = VertexBuffers[draw.VertexSlot].Vertices[i2 + draw.VertexOffset].Position;
    const vec3 p3 = VertexBuffers[draw.VertexSlot].Vertices[i3 + draw.VertexOffset].Position;

    // Transform to view space (Blender's approach uses view-space face normals)
    const vec3 vs0 = (MV * vec4(p0, 1.0)).xyz;
    const vec3 vs1 = (MV * vec4(p1, 1.0)).xyz;
    const vec3 vs2 = (MV * vec4(p2, 1.0)).xyz;
    const vec3 vs3 = (MV * vec4(p3, 1.0)).xyz;

    // View vector: perspective = toward edge_v0, ortho = (0,0,-1)
    const vec3 view_vec = normalize(vs1);

    // Edge vectors from edge_v0
    const vec3 v10 = vs0 - vs1; // toward adj_left
    const vec3 v12 = vs2 - vs1; // toward edge_v1
    const vec3 v13 = vs3 - vs1; // toward adj_right

    // Face normals of the two triangles sharing this edge.
    // cross(v10, v12) gives outward-pointing normal matching our face winding convention.
    vec3 n0 = cross(v10, v12); // left face
    float len_n0 = length(n0);
    n0 = len_n0 > 0.0 ? n0 / len_n0 : vec3(0);
    vec3 n3 = cross(v12, v13); // right face
    float len_n3 = length(n3);
    n3 = len_n3 > 0.0 ? n3 / len_n3 : vec3(0);

    float fac0 = dot(view_vec, n0);
    float fac3 = dot(view_vec, n3);

    // If one face is perpendicular to view, consider it an outline edge.
    // Otherwise, if both face the camera the same way, it's internal → discard.
    if (abs(fac0) > 1e-5 && abs(fac3) > 1e-5) {
        if (sign(fac0) == sign(fac3)) {
            gl_Position = vec4(0, 0, -2, 1);
            Color = vec4(0);
            EdgeStart = vec2(0);
            EdgePos = vec2(0);
            return;
        }
    }

    // Concavity check: don't outline concave edges
    bool inverted = dot(cross(M[0].xyz, M[1].xyz), M[2].xyz) < 0.0;
    vec3 n0_check = inverted ? -n0 : n0;
    if (dot(n0_check, v13) > 0.0001) {
        gl_Position = vec4(0, 0, -2, 1);
        Color = vec4(0);
        EdgeStart = vec2(0);
        EdgePos = vec2(0);
        return;
    }

    // Choose which endpoint — compute depth identically to BoneSolid.vert
    // (extract .xyz then repack with w=1) to ensure bit-identical depth values.
    vec3 world_pos = (M * vec4((vert_in_edge == 0u) ? p1 : p2, 1.0)).xyz;
    vec4 clip_pos = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);

    Color = vec4(bone_wire_color(load_bone_instance_state(draw)), 1.0);

    // Depth bias: push wire in front of fill to avoid z-fighting (matches Blender).
    clip_pos.z -= 1e-4;

    gl_Position = clip_pos;
    const vec2 screen_pos = (clip_pos.xy / clip_pos.w * 0.5 + 0.5) * SceneViewUBO.ViewportSize;
    EdgeStart = screen_pos;
    EdgePos = screen_pos;
}
