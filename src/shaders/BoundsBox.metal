#ifndef BOUNDSBOX_MSL
#define BOUNDSBOX_MSL

// One bounding-box wireframe per instance, reading the instance-arena slot list, local bounds, and
// transform directly. 24 vertices per instance, two per box edge.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "AABB.metal"
#include "BoundsBoxPushConstants.metal"
#include "BoxWire.metal"

// Each threadgroup emits up to 6 boxes, one thread per emitted vertex.
constant uint BoundsBoxCount = 6u;
constant uint BoundsBoxVertices = 24u;
using BoundsBoxOutput = metal::mesh<LineVaryings, void, BoundsBoxCount * BoundsBoxVertices, BoundsBoxCount * BoundsBoxVertices / 2u, metal::topology::line>;

// One box-edge endpoint, in the local bounds of the instance at `slot`.
inline LineVaryings BoundsBoxCorner(const thread Scene &scene, constant BoundsBoxPushConstants &pc, uint slot, uint vertex_id) {
    const AABB aabb = BindlessBuffer(AABB, scene.B.Buffer, pc.BoundsSlot)[slot];

    LineVaryings out;
    out.Color = float4(0);
    out.EdgeStart = float2(0);
    out.EdgePos = float2(0);
    const float3 aabb_min = float3(aabb.Min), aabb_max = float3(aabb.Max);
    if (any(aabb_min > aabb_max)) {
        out.Position = float4(2, 2, 2, 1); // Empty bounds: place the vertex outside the clip volume.
        return out;
    }
    const uint corner = BoxEdgeCorners[vertex_id];
    const float3 local = mix(aabb_min, aabb_max, float3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u)));
    const Transform world = scene.Models(pc.ModelSlot)[slot];
    const float3 world_pos = trs_transform_point(world, local);

    // Boxes draw only for selected instances.
    const uint instance_state = uint(scene.InstanceStates(pc.StateSlot)[slot]);
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
    out.Color = float4(is_active ? float3(scene.Theme.Colors.ObjectActive) : float3(scene.Theme.Colors.ObjectSelected), 1.0f);

    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    const float2 screen_pos = clip_to_frag_co(out.Position, float2(scene.View.ViewportSize));
    out.EdgeStart = screen_pos;
    out.EdgePos = screen_pos;
    return out;
}

[[mesh]] void BoundsBoxMesh(
    BoundsBoxOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoundsBoxPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint first_box = threadgroup_position.x * BoundsBoxCount;
    const uint box_count = min(BoundsBoxCount, pc.BoxCount - first_box);
    output.set_primitive_count(box_count * BoundsBoxVertices / 2u);
    if (thread_index >= box_count * BoundsBoxVertices) return;

    const uint box = first_box + thread_index / BoundsBoxVertices;
    const uint slot = BindlessBuffer(uint, bindless.Buffer, pc.SlotsSlot)[box];
    output.set_vertex(thread_index, BoundsBoxCorner(scene, pc, slot, thread_index % BoundsBoxVertices));
    output.set_index(thread_index, thread_index);
}

#endif
