// EEVEE GeometryVelocity adapted to MeshEditor's evaluated position arenas.
#pragma once
#include "VisibilityDecode.metal"
#include "gpu/RayTracingRecord.h"
inline RayDeformedSurface RayPreviousVisibleSurface(
    uint2 pixel, const thread ResolvedVisibility &resolved, device const BindlessSet &bindless,
    constant SceneViewUBO &view, constant ViewportThemeUbo &theme, constant WorkspaceLights &workspace,
    const thread RayMotionTransform &transform, device const packed_float3 *previous_positions
) {
    const auto draw = resolved.Draw;
    if (MeshletCoarse(resolved.Meshlet)) return {};
    const Scene scene{bindless, view, theme, workspace};
    const Transform world = MeshletWorld(scene, draw);
    const uint3 corners = MeshletCornerIds(bindless, resolved.Payloads, resolved.Meshlet, draw, resolved.Triangle, resolved.LocalTriangle);
    float4 clips[3];
    float3 old[3];
    for (uint i = 0; i < 3; ++i) {
        const uint vertex_id = scene.CornerVertex(draw, corners[i]);
        const uint address = scene.VertexAddress(draw, vertex_id);
        const uint posed = PoseAttributeIndex(bindless, view.PosedPositionNodesSlot, draw.PositionNamespace, address);
        if (posed == InvalidOffset) return {};
        const float3 current = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, vertex_id)));
        clips[i] = scene.ViewProj() * float4(current, 1);
        const uint previous_block = reinterpret_cast<device const uint *>(previous_positions)[posed >> 8u];
        if (previous_block == InvalidOffset) return {};
        const float3 p = apply_object_pending_transform(scene, draw, trs_transform_point(world, float3(previous_positions[previous_block + (posed & 255u)])));
        old[i] = (transform.CurrentToPrevious.Unpack() * float4(p, 1)).xyz;
    }
    const float3 weights = TriangleWeights(float2(pixel) + .5f, clips[0], clips[1], clips[2], float2(view.ViewportSize)).Value;
    const float3 position = PerspectiveValue(weights, old[0], old[1], old[2]);
    float3 normal = cross(old[1] - old[0], old[2] - old[0]);
    const float magnitude = length(normal);
    if (!(magnitude > 1e-20f) || !all(isfinite(position)) || !all(isfinite(normal))) return {};
    return {float4(position, 1), float4(normal / magnitude, 0)};
}
