#include "ElementWorkShared.metal"
#include "ElementMembershipRead.metal"

kernel void ElementWorkSeed(JOB_TABLE_ARGS(ElementWorkSeedJob)) {
    JOB_TILE(job);
    const uint element = MembershipElement(b,job,tile,lane);
    if (element != InvalidOffset) MarkWork(b,job.Work,element);
}
