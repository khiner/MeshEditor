#ifndef SELECTIONELEMENTLINKEDLIST_MSL
#define SELECTIONELEMENTLINKEDLIST_MSL

#include "SelectionShared.metal"
#include "Varyings.metal"
#include "SelectionElementPushConstants.metal"

// The element selection pass appends every covered fragment to its pixel's list. It writes no color.
[[early_fragment_tests]]
fragment void SelectionElementLinkedListFragment(
    ElementIdVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    SelectionAppend(scene, pc.Selection, uint2(in.Position.xy), in.Position.z, in.ElementId);
}

#endif
