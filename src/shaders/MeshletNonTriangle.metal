#ifndef MESHLET_NON_TRIANGLE_MSL
#define MESHLET_NON_TRIANGLE_MSL

#include "LineQuad.metal"
#include "MeshletResolve.metal"
#include "SceneUBO.metal"

constant float2 PointQuadCorners[4] = {
    float2(-1.0f, 1.0f), float2(-1.0f, -1.0f), float2(1.0f, 1.0f), float2(1.0f, -1.0f)
};

inline float4 PointQuadPosition(const thread Scene &scene, float4 clip, uint corner) {
    clip.xy += PointQuadCorners[corner] * (PointSize * 0.5f) / float2(scene.View.ViewportSize) * 2.0f * clip.w;
    return clip;
}

inline uint NonTriangleVertexId(
    device const BindlessSet &bindless, uint meshlet_vertex_slot, MeshletRecord meshlet,
    uint topology, uint element, uint corner
) {
    const uint endpoint = topology == MeshPrimitiveTopology_Line ? line_quad_endpoint(corner) : 0u;
    const uint stride = topology == MeshPrimitiveTopology_Line ? 2u : 1u;
    return MeshletPackedVertex(bindless, meshlet_vertex_slot, meshlet, element * stride + endpoint);
}

inline float4 NonTrianglePosition(
    const thread Scene &scene, device const BindlessSet &bindless, uint meshlet_vertex_slot,
    DrawData draw, MeshletRecord meshlet, uint topology, uint element, uint corner
) {
    const Transform world = MeshletWorld(scene, draw);
    const uint source_vertex = NonTriangleVertexId(bindless, meshlet_vertex_slot, meshlet, topology, element, corner);
    const float4 clip = MeshletPosition(scene, draw, world, source_vertex);
    if (topology == MeshPrimitiveTopology_Point) return PointQuadPosition(scene, clip, corner);
    const uint other = NonTriangleVertexId(bindless, meshlet_vertex_slot, meshlet, topology, element, corner ^ 2u);
    const float4 other_clip = MeshletPosition(scene, draw, world, other);
    const bool first = line_quad_endpoint(corner) == 0u;
    return line_quad_position(
        scene, first ? clip : other_clip, first ? other_clip : clip, corner, scene.Theme.EdgeWidth + 0.5f
    );
}

#endif
