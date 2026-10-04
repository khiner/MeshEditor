#ifndef OBJECTORIGIN_MSL
#define OBJECTORIGIN_MSL

#include "Bindless.metal"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/ObjectOriginPushConstants.h"

struct ObjectOriginVaryings {
    float4 Position [[position]];
    float2 Offset [[user(Offset)]]; // From the dot's center, in render pixels.
    float3 Color [[user(Color)]] [[flat]];
};

constant float2 OriginCorners[4] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)};

// A screen-aligned quad around each visible selected or active object's origin, and a point outside the clip volume for every other slot.
// Bones and joints draw their own selection, so they take no dot.
vertex ObjectOriginVaryings ObjectOriginVertex(
    uint vertex_id [[vertex_id]],
    uint slot [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ObjectOriginPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const ObjectOriginVaryings culled{float4(2.0f, 2.0f, 2.0f, 1.0f), float2(0.0f), float3(0.0f)};
    const uint state = uint(scene.InstanceStates(view.InstanceStateSlot)[slot]);
    if ((state & (STATE_SELECTED | STATE_ACTIVE)) == 0u || (state & STATE_HIDDEN) != 0u) return culled;
    // An instance without a mesh record, like an empty's, draws its dot.
    const uint mesh = scene.InstanceRecords(view.InstanceRecordSlot)[slot].Mesh;
    if (mesh != InvalidOffset && (scene.MeshRecords(view.MeshRecordSlot)[mesh].Display.Flags & uint(MeshletInstanceFlag::OverlayOnly)) != 0u) return culled;
    const float4 clip = scene.ViewProj() * float4(float3(scene.Models(pc.TransformSlot)[slot].P), 1.0f);
    if (clip.w <= 0.0f) return culled;
    // One pixel past the rim leaves room for its antialiased edge.
    const float2 offset = OriginCorners[vertex_id] * (pc.RadiusPx + 1.0f);
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    return {
        float4(clip.xy / clip.w + offset * 2.0f / float2(view.ViewportSize), 0.0f, 1.0f),
        offset,
        (state & STATE_ACTIVE) != 0u ? float3(colors.ObjectActive) : float3(colors.ObjectSelected),
    };
}

// A filled disc in the object's selection color inside a dark rim, premultiplied.
fragment float4 ObjectOriginFragment(
    ObjectOriginVaryings in [[stage_in]],
    constant ObjectOriginPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const float distance = length(in.Offset);
    const float coverage = saturate(pc.RadiusPx + 0.5f - distance);
    if (coverage <= 0.0f) discard_fragment();
    const float fill = saturate(pc.RadiusPx - pc.OutlinePx + 0.5f - distance);
    return float4(in.Color * fill * coverage, coverage);
}

#endif
