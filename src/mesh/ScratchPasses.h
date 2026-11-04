#pragma once
#include "gpu/Collider.h"
#include "mesh/MeshPipelines.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"

// A run of `words` in the chain's scratch, which the GPU addresses with 32-bit offsets.
Range TakeScratch(mtl::ComputeChain &, uint64_t words);
void FillScratch(mtl::ComputeChain &, const MeshPipelines &, uint32_t run, uint32_t count, uint32_t value);
// Scans the `length` values at `values` in the chain's scratch in place, leaving their sum in the terminator that ends them.
void ScanScratch(mtl::ComputeChain &, const MeshPipelines &, uint32_t values, uint32_t length);
template<typename T> std::vector<T> ReadScratch(const mtl::ComputeChain &chain, uint32_t offset, size_t count) {
    const auto words = chain.Scratch.Get({offset, uint32_t(count * sizeof(T) / sizeof(uint32_t))});
    std::vector<T> values(count);
    std::memcpy(values.data(), words.data(), sizeof(T) * count);
    return values;
}
// Clears the word at `changed` in the chain's scratch, records `pass` over `threads` threads and submits, until a round leaves the word zero.
uint32_t RunUntilUnchanged(mtl::ComputeChain &chain, const mtl::ComputePipeline &pass, const auto &pc, uint32_t threads, uint32_t changed) {
    uint32_t rounds = 0u;
    do {
        chain.Scratch.GetMutable({changed, 1u})[0] = 0u;
        chain.Threads(pass, pc, threads);
        chain.Submit();
        ++rounds;
    } while (chain.Scratch.Get({changed, 1u})[0]);
    return rounds;
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
// Records the key sort and the tree build of the `layers` consecutive colliders whose runs start at `runs`, each over `leaf_groups` threadgroups.
void RecordColliderBuild(mtl::ComputeChain &, const MeshPipelines &, SlotOffset runs, uint32_t leaf_groups, uint32_t layers);

// The pointer-jumping rounds that label and rank every list over `length` elements, one per doubling of the longest list.
constexpr uint32_t LoopRounds(uint32_t length) { return uint32_t(std::bit_width(length)) + 1u; }
// Records the label, rank start and rank rounds of `passes`, the count over `length` elements with a terminator, and in-place scans of `lengths` and `flags`.
// Returns `pc` with Parameter holding the final buffers' parity, for the output pass.
auto RecordLoopRanks(mtl::ComputeChain &chain, const MeshPipelines &pipelines, auto pc, std::array<MeshPass, 4> passes, uint32_t length, uint32_t threads, uint32_t lengths, uint32_t flags) {
    const uint32_t rounds = LoopRounds(length), parity = rounds & 1u;
    for (pc.Parameter = 0u; pc.Parameter < rounds; ++pc.Parameter) chain.Threads(pipelines[passes[0]], pc, threads);
    pc.Parameter = parity;
    chain.Threads(pipelines[passes[1]], pc, threads);
    for (pc.Parameter = 0u; pc.Parameter < rounds; ++pc.Parameter) chain.Threads(pipelines[passes[2]], pc, threads);
    pc.Parameter = parity;
    chain.Threads(pipelines[passes[3]], pc, length + 1u);
    ScanScratch(chain, pipelines, lengths, length + 1u);
    ScanScratch(chain, pipelines, flags, length + 1u);
    return pc;
}
