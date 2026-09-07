#ifndef SELECTIONFRAGMENT_MSL
#define SELECTIONFRAGMENT_MSL

#include "SelectionObjectQuery.metal"
#include "EdgeQuad.metal"
#include "ObjectSelectionPushConstants.metal"

// Accumulates each covered fragment into the object-selection query.
fragment void SelectionFragment(
    ObjectIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant ObjectSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    WriteObjectSelect(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ObjectId);
}

fragment void SelectionStrokeFragment(
    EdgeQuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ObjectSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    ShadeEdgeQuad(in, Scene{bindless, view, theme, workspace}, false);
    WriteObjectSelect(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ObjectId);
}

#endif
