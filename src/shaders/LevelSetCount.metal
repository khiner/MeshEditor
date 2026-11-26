#include "Lattice.metal"
#include "gpu/LevelSetCountJob.h"

kernel void LevelSetCount(JOB_TABLE_ARGS(LevelSetCountJob)) {
    JOB_THREAD(job, v);
    const auto values = [&b, &job](uint p) { return as_type<float>(ReadNamedAttribute(b, job.Values, AttributeType::Float, p).x); };
    const LatticeField<decltype(values)> field{int3(int(job.Size[0]), int(job.Size[1]), int(job.Size[2])), values, job.Level};
    const uint2 counts = v < job.Points ? LatticeCounts(field, LatticeAt(field.Size, v)) : uint2(0u);
    const uint2 total = simd_sum(counts);
    if (simd_is_first() && any(total != 0u)) {
        device uint *words = BindlessBufferMutable(uint, b.Buffer, job.Counts.Slot) + job.Counts.Offset;
        atomic_fetch_add_explicit(AtomicWord(words), total.x, memory_order_relaxed);
        atomic_fetch_add_explicit(AtomicWord(words + 1u), total.y, memory_order_relaxed);
    }
}
