#ifndef SELECTIONELEMENTPICK_MSL
#define SELECTIONELEMENTPICK_MSL

#include "SelectionPickKey.metal"
#include "Varyings.metal"
#include "SelectionElementPushConstants.metal"

// Resolves point, line, and X-ray face IDs through one selection key.
[[early_fragment_tests]]
fragment void SelectionElementPickFragment(
    ElementIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    WriteElementPick(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ElementId);
}

#endif
