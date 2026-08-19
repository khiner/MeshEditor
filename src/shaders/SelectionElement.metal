#ifndef SELECTIONELEMENT_MSL
#define SELECTIONELEMENT_MSL

// The three element selection vertex stages: one per element kind, each tagging its fragments with
// the element id the pick and box passes read back.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "SelectionElementPushConstants.metal"

vertex ElementIdVaryings SelectionElementFaceVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    ElementIdVaryings out;
    out.ElementId = draw.ElementIdOffset + scene.ObjectIds(draw.ObjectIdSlot)[draw.FaceIdOffset + vertex_id / 3u];
    out.Position = scene.ViewProj() * float4(trs_transform_point(world, float3(vert.Position)), 1.0f);
    out.PointSize = 1.0f; // Point topology catches edge-on X-Ray face hits.
    return out;
}

vertex ElementIdVaryings SelectionElementEdgeVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    ElementIdVaryings out;
    out.ElementId = draw.ElementIdOffset + vertex_id / 2u + 1u;
    out.Position = scene.ViewProj() * float4(trs_transform_point(world, float3(vert.Position)), 1.0f);
    // A slightly enlarged point fallback reduces sample-center misses on near-zero-length projected edges.
    out.PointSize = 2.0f;
    return out;
}

vertex ElementIdVaryings SelectionElementVertexVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint idx = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_id];

    // A set element-state slot filters to selected vertices alone.
    if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        const uint state = uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + idx]);
        if ((state & STATE_SELECTED) == 0u) {
            // Clip non-selected vertices by placing them outside the frustum.
            return ElementIdVaryings{float4(0, 0, 0, 0), 0.0f, 0u};
        }
    }

    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    ElementIdVaryings out;
    out.ElementId = draw.ElementIdOffset + idx + 1u;
    out.Position = scene.ViewProj() * float4(trs_transform_point(world, float3(vert.Position)), 1.0f);
    out.PointSize = PointSize;
    return out;
}

#endif
