#ifndef BOUNDSBOX_MSL
#define BOUNDSBOX_MSL

// One bounding-box wireframe per instance, reading the instance-arena slot list, local bounds, and
// transform directly. 24 vertices per instance, two per box edge.
#include "Bindless.metal"
#include "Varyings.metal"
#include "AABB.metal"
#include "BoundsBoxPushConstants.metal"

// Corner indices for the 12 box edges. Corner bits select Min/Max per axis: x=1, y=2, z=4.
constant uint EdgeCorners[24] = {
    0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u, // bottom ring
    4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u, // top ring
    0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u  // verticals
};

vertex LineVaryings BoundsBoxVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoundsBoxPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint slot = BindlessBuffer(uint, bindless.Buffer, pc.SlotsSlot)[instance_id];
    const AABB aabb = BindlessBuffer(AABB, bindless.Buffer, pc.BoundsSlot)[slot];

    LineVaryings out;
    out.Color = float4(0);
    out.EdgeStart = float2(0);
    out.EdgePos = float2(0);
    const float3 aabb_min = float3(aabb.Min), aabb_max = float3(aabb.Max);
    if (any(aabb_min > aabb_max)) {
        out.Position = float4(2, 2, 2, 1); // Empty bounds: place the vertex outside the clip volume.
        return out;
    }
    const uint corner = EdgeCorners[vertex_id];
    const float3 local = mix(aabb_min, aabb_max, float3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u)));
    const Transform world = scene.Models(pc.ModelSlot)[slot];
    const float3 world_pos = trs_transform_point(world, local);

    // Boxes draw only for selected instances.
    const uint instance_state = uint(scene.InstanceStates(pc.StateSlot)[slot]);
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
    out.Color = float4(is_active ? float3(scene.Theme.Colors.ObjectActive) : float3(scene.Theme.Colors.ObjectSelected), 1.0f);

    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    const float2 screen_pos = clip_to_frag_co(out.Position, float2(view.ViewportSize));
    out.EdgeStart = screen_pos;
    out.EdgePos = screen_pos;
    return out;
}

#endif
