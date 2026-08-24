#ifndef SELECTIONFRAGMENT_MSL
#define SELECTIONFRAGMENT_MSL

#include "SelectionObjectQuery.metal"
#include "Varyings.metal"
#include "SelectionDrawPushConstants.metal"

// The object selection pass folds every covered fragment into the query's keys and bits. It writes no color.
fragment void SelectionFragment(
    ObjectIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    WriteObjectSelect(bindless, pc.Query, uint2(in.Position.xy), in.Position.z, in.ObjectId);
}

#endif
