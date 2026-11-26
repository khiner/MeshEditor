#include "mesh/ScratchPasses.h"

#include <ranges>

Range TakeScratch(mtl::ComputeChain &chain, uint64_t words) {
    if (words > UINT32_MAX) throw std::length_error("GPU scratch exceeds its 32-bit address space.");
    return chain.Scratch.Allocate(uint32_t(words));
}

void FillScratch(mtl::ComputeChain &chain, const MeshPipelines &pipelines, std::span<const ScratchFillRun> runs) {
    mtl::GatherPass(chain, pipelines[MeshPass::ScratchFill], runs, &ScratchFillRun::Count);
}

void RecordColliderBuild(mtl::ComputeChain &chain, const MeshPipelines &pipelines, std::span<const ColliderBuild> colliders) {
    const auto records = colliders | std::views::transform([&](const ColliderBuild &collider) { return SlotOffset{chain.Scratch.Buffer.Slot, collider.Runs}; }) | std::ranges::to<std::vector>();
    // Sorting and tree passes take a tile per JobTileThreads leaves, and prefix passes a tile per digit.
    const auto leaf_tiles = colliders | std::views::transform([](const ColliderBuild &collider) { return TileCount(collider.Leaves, JobTileThreads); }) | std::ranges::to<std::vector>();
    const std::vector<uint32_t> digit_tiles(colliders.size(), 16u);
    // Inside a lane the sort records once the lanes record, so it holds its records and tile counts.
    mtl::GatherJobs(chain, &pipelines[MeshPass::ColliderSortHistogram], std::span{records}, leaf_tiles, [&chain, &pipelines, records, digit_tiles](const mtl::JobTable &leaves, auto) {
        mtl::GatherJobs(chain, &pipelines[MeshPass::ColliderSortPrefix], std::span{records}, digit_tiles, [&](const mtl::JobTable &digits, auto) {
            const auto dispatch = [&](MeshPass pass, mtl::JobTable table, uint32_t shift) {
                table.Constants.Parameter = shift;
                chain.Jobs(pipelines[pass], table, JobTileThreads);
            };
            for (uint32_t shift = 0u; shift < 32u; shift += 4u) {
                dispatch(MeshPass::ColliderSortHistogram, leaves, shift);
                dispatch(MeshPass::ColliderSortPrefix, digits, shift);
                dispatch(MeshPass::ColliderSortScatter, leaves, shift);
            }
            dispatch(MeshPass::ColliderTree, leaves, 0u);
        });
    });
}

void ScanScratch(mtl::ComputeChain &chain, const MeshPipelines &pipelines, std::span<const Range> runs) {
    const auto blocks = [&](const Range &run) { return TileCount(run.Count, BlockElements); };
    std::vector<ScratchScanRun> scans;
    for (const auto &run : runs) scans.push_back({run.Offset, run.Count, TakeScratch(chain, blocks(run)).Offset});
    // The sum and offset passes take a tile per scan block of a run, and the prefix pass a tile per run.
    const auto block_tiles = runs | std::views::transform(blocks) | std::ranges::to<std::vector>();
    const std::vector<uint32_t> run_tiles(runs.size(), 1u);
    mtl::GatherPass(chain, pipelines[MeshPass::ScratchScanSum], std::span{scans}, block_tiles, JobTileThreads);
    mtl::GatherPass(chain, pipelines[MeshPass::ScratchScanPrefix], std::span{scans}, run_tiles, JobTileThreads);
    mtl::GatherPass(chain, pipelines[MeshPass::ScratchScanOffsets], std::span{scans}, block_tiles, JobTileThreads);
}
