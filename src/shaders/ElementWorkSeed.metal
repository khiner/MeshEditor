#include "ElementWorkShared.metal"
#include "ElementMembershipRead.metal"

kernel void ElementWorkSeed(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ElementWorkSeedJob &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const uint element = MembershipElement(b,pc,group,lane);
    if (element != InvalidOffset) MarkWork(b,pc.Work,element);
}
