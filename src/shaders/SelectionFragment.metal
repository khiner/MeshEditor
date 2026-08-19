#ifndef SELECTIONFRAGMENT_MSL
#define SELECTIONFRAGMENT_MSL

#include "SelectionShared.metal"
#include "Varyings.metal"
#include "SelectionDrawPushConstants.metal"

// The object selection pass appends every covered fragment to its pixel's list. It writes no color.
fragment void SelectionFragment(
    ObjectIdVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SelectionDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    SelectionAppend(scene, pc.Selection, uint2(in.Position.xy), in.Position.z, in.ObjectId);
}

#endif
