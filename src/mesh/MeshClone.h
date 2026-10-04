#pragma once

#include "gpu/MeshCloneJob.h"

#include <vector>

struct MeshPipelines;
namespace mtl {
struct Buffer;
struct ComputeChain;
} // namespace mtl

// The GPU work of a batch of clones: byte runs copied and records gathered by rank within their buffers, then uint32 references rebased in the copies by delta or by rank and reference pairs copied with deltas.
// Each buffer's runs of one kind take one dispatch, so a batch costs a fixed number of dispatches per buffer at any clone count.
// A copy, rank rebase or pair copy that continues its buffer's previous one of its kind extends that run.
struct CloneCopies {
    void Copy(mtl::Buffer &, uint64_t source, uint64_t destination, uint64_t bytes);
    // Adds `delta` to `count` references `stride` words apart from `byte_offset`, keeping the null sentinel.
    void Rebase(mtl::Buffer &, uint64_t byte_offset, uint32_t count, uint32_t delta, uint32_t stride = 1u);
    // Copies the records of `index`'s members in rank order, `bytes` each, to the `count` records from `destination`.
    void GatherByRank(mtl::Buffer &, uint32_t destination, uint32_t count, uint32_t bytes, MeshletIndexRef index);
    // Replaces `count` handles `stride` words apart from `byte_offset` by `first` plus their rank among `index`'s members, keeping the null sentinel.
    void RebaseByRank(mtl::Buffer &, uint64_t byte_offset, uint32_t count, MeshletIndexRef index, uint32_t first, uint32_t stride = 1u);
    // Copies `count` uint32 pairs from pair `source` to pair `destination`, adding the deltas to their non-null members.
    void CopyPairs(mtl::Buffer &, uint32_t source, uint32_t destination, uint32_t count, uint32_t first_delta, uint32_t second_delta);
    // Records every copy and gather, then every rebase and pair copy, on the chain, which retains the job tables.
    void Encode(mtl::ComputeChain &, const MeshPipelines &);

    // Each buffer's jobs of one kind.
    template<typename Job> struct Runs {
        std::vector<mtl::Buffer *> Buffers;
        std::vector<std::vector<Job>> Jobs;
        void Add(mtl::Buffer &, const Job &);
    };

private:
    Runs<ByteCopy> Copies;
    Runs<RankGather> Gathers;
    Runs<IndexRebase> Rebases;
    Runs<RankRebase> RankRebases;
    Runs<ReferencePairCopy> Pairs;
};
