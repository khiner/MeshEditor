#ifndef SELECTIONELEMENTPICK_MSL
#define SELECTIONELEMENTPICK_MSL

#include "SelectionPickKey.metal"
#include "Varyings.metal"
#include "gpu/SelectionElementPushConstants.h"

[[early_fragment_tests]]
fragment void SelectionElementPickFragment(
    ElementIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    WriteElementPick(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ElementId);
}

[[early_fragment_tests]]
fragment void SelectionFacePickFragment(
    FaceIdFragmentVaryings in [[stage_in]], bool front_facing [[front_facing]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (FaceCulled(in.Facing, front_facing)) return;
    WriteElementPick(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ElementId);
}

#endif
