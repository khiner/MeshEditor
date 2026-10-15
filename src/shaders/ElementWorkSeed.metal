#include "ElementWorkShared.metal"
#include "ElementMembershipRead.metal"
#include "gpu/TiledJobPushConstants.h"

// Each tile reads one listed block of its job, and each job fills its own work.
kernel void ElementWorkSeed(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    device const uint *storage = BindlessBuffer(uint,b.Buffer,pc.StorageSlot);
    const uint2 tile = reinterpret_cast<device const uint2 *>(storage+pc.TileMapOffset)[pc.FirstTile+group];
    const ElementWorkSeedJob job = reinterpret_cast<device const ElementWorkSeedJob *>(storage+pc.JobsOffset)[tile.x];
    const uint element = MembershipElement(b,job,tile.y,lane);
    if (element != InvalidOffset) MarkWork(b,job.Work,element);
}
