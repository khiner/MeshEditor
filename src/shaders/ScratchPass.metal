#include "BlockScan.metal"
#include "Bindless.metal"
#include "gpu/ScratchPassJob.h"

// Fills runs of the scratch, one tile per JobTileThreads words of a run.
kernel void ScratchFill(JOB_TABLE_ARGS(ScratchFillRun)) {
    JOB_THREAD(fill, i);
    if (i < fill.Count) ScratchOf(b, table).Run(fill.Values)[i] = fill.Value;
}

// Exclusive scans of runs of values in place, each run's sum landing in the entry past it when the run ends with a zero.
// The sum and offset passes take one tile per scan block of a run, and the prefix pass one tile per run.
kernel void ScratchScanSum(JOB_TABLE_ARGS(ScratchScanRun), uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    JOB_TILE(run);
    const JobRuns runs = ScratchOf(b, table);
    threadgroup uint sums[ScanSimdGroups + 1u];
    ScanBlockSum(runs.Run(run.Values), run.Count, tile, runs.Run(run.Blocks), lane, simd_lane, simd_group, sums);
}
kernel void ScratchScanPrefix(JOB_TABLE_ARGS(ScratchScanRun), uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    JOB_TILE(run);
    threadgroup uint sums[ScanSimdGroups + 1u];
    ScanBlockPrefix(ScratchOf(b, table).Run(run.Blocks), (run.Count + ScanBlockElements - 1u) / ScanBlockElements, lane, simd_lane, simd_group, sums);
}
kernel void ScratchScanOffsets(JOB_TABLE_ARGS(ScratchScanRun), uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    JOB_TILE(run);
    const JobRuns runs = ScratchOf(b, table);
    threadgroup uint sums[ScanSimdGroups + 1u];
    device uint *values = runs.Run(run.Values);
    ScanBlockOffsets(values, run.Count, tile, runs.Run(run.Blocks), values, lane, simd_lane, simd_group, sums);
}
