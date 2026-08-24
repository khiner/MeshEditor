#ifndef SELECTIONELEMENTPICK_MSL
#define SELECTIONELEMENTPICK_MSL

#include "SelectionPickKey.metal"
#include "Varyings.metal"
#include "SelectionElementPushConstants.metal"

// Element picking from the id raster: points, lines and X-ray faces all resolve through one key.
[[early_fragment_tests]]
fragment void SelectionElementPickFragment(
    ElementIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    WriteElementPick(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ElementId);
}

#endif
