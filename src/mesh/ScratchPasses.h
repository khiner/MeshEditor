#pragma once
#include "gpu/Collider.h"
#include "gpu/ScratchPassJob.h"
#include "mesh/MeshPipelines.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"

// A run of `words` in the chain's scratch, which the GPU addresses with 32-bit offsets.
Range TakeScratch(mtl::ComputeChain &, uint64_t words);
// Fills each run of the chain's scratch with its value.
void FillScratch(mtl::ComputeChain &, const MeshPipelines &, std::span<const ScratchFillRun>);
// Scans each run of values in the chain's scratch in place, leaving its sum in the terminator that ends it.
void ScanScratch(mtl::ComputeChain &, const MeshPipelines &, std::span<const Range> runs);
template<typename T> std::vector<T> ReadScratch(const mtl::ComputeChain &chain, uint32_t offset, size_t count) {
    const auto words = chain.Scratch.Get({offset, uint32_t(count * sizeof(T) / sizeof(uint32_t))});
    std::vector<T> values(count);
    std::memcpy(values.data(), words.data(), sizeof(T) * count);
    return values;
}
// Records rounds of `record()` in the lane of `key`, each after clearing the word at `changed` in the chain's scratch, and calls `then` once the chain completed a round that leaves the word zero.
inline void RecordUntilUnchanged(mtl::ComputeChain &chain, uint32_t key, uint32_t changed, std::function<void()> record, std::function<void()> then) {
    chain.Scratch.GetMutable({changed, 1u})[0] = 0u;
    chain.Lane(key, record);
    chain.AfterSubmit([&chain, key, changed, record = std::move(record), then = std::move(then)] mutable {
        if (chain.Scratch.Get({changed, 1u})[0]) return RecordUntilUnchanged(chain, key, changed, std::move(record), std::move(then));
        then();
    });
}

ColliderRuns AllocateCollider(auto &&alloc, uint32_t leaves) {
    const uint64_t blocks = TileCount(leaves, TileElements), internal = std::max(leaves, 2u) - 1u;
    return {
        .Leaves = leaves,
        .Keys = alloc(leaves),
        .Order = alloc(leaves),
        .Temporary = alloc(leaves),
        .Histogram = alloc(16u * blocks),
        .Totals = alloc(16u),
        .NodeBoxes = alloc(6u * (2ull * leaves - 1u)),
        .NodeParents = alloc(2ull * leaves - 1u),
        .Children = alloc(2u * internal),
        .Counters = alloc(internal),
    };
}
// A collider over Leaves boxes whose runs lie at word Runs of the chain's scratch.
struct ColliderBuild {
    uint32_t Runs, Leaves;
};
// Records the key sort and the tree build of the colliders.
void RecordColliderBuild(mtl::ComputeChain &, const MeshPipelines &, std::span<const ColliderBuild>);

// A job's lists for its loop ranks: their element count, and its runs of lengths and flags, each a word per element then a terminator.
struct LoopList {
    uint32_t Length, Lengths, Flags;
};
// Records the loop ranks of `job` and the jobs the other pending lanes gather for `passes` at the same step, or of `job` at once outside a lane.
// The label rounds, rank start and rank rounds take each job's `tiles` threadgroups, with a pointer-jumping round per doubling of the longest list.
// The count then takes one thread per element of its list and one for the terminator, with the final buffers' parity, and the in-place scans of the lengths and flags follow.
// `list(job)` gives a job's LoopList.
template<typename Job> void GatherLoopRanks(mtl::ComputeChain &chain, const MeshPipelines &pipelines, std::array<MeshPass, 4> passes, const Job &job, uint32_t tiles, auto list) {
    mtl::GatherJobs(chain, &pipelines[passes[0]], std::span{&job, 1u}, std::array{tiles}, [&chain, &pipelines, passes, list](const mtl::JobTable &blocks, std::span<const Job> jobs) {
        uint32_t longest = 0u;
        std::vector<uint32_t> count_tiles;
        std::vector<Range> scans;
        for (const auto &gathered : jobs) {
            const LoopList lists = list(gathered);
            longest = std::max(longest, lists.Length);
            count_tiles.push_back(TileCount(uint64_t(lists.Length) + 1u, JobTileThreads));
            scans.insert(scans.end(), {Range{lists.Lengths, lists.Length + 1u}, Range{lists.Flags, lists.Length + 1u}});
        }
        mtl::GatherJobs(chain, &pipelines[passes[3]], jobs, count_tiles, [&](const mtl::JobTable &counts, auto) {
            const auto record = [&](MeshPass pass, uint32_t parameter) {
                auto table = pass == passes[3] ? counts : blocks;
                table.Constants.Parameter = parameter;
                chain.Jobs(pipelines[pass], table, JobTileThreads);
            };
            const uint32_t rounds = uint32_t(std::bit_width(longest)) + 1u, parity = rounds & 1u;
            for (uint32_t round = 0u; round < rounds; ++round) record(passes[0], round);
            record(passes[1], parity);
            for (uint32_t round = 0u; round < rounds; ++round) record(passes[2], round);
            record(passes[3], parity);
            ScanScratch(chain, pipelines, scans);
        });
    });
}
