#pragma once

#include "Range.h"
#include "gpu/MeshCloneJob.h"

#include "numeric/uvec2.h"
#include <span>
#include <vector>

struct MeshPipelines;
namespace mtl {
struct Buffer;
struct ComputeChain;
} // namespace mtl

// A clone batch copies owned byte runs and gathers render records, then remaps their references.
// Each buffer's runs of one kind take one dispatch, so a batch costs a fixed number of dispatches per buffer at any clone count.
// Contiguous copies and compatible remapping jobs coalesce into runs.
struct CloneCopies {
    void Copy(mtl::Buffer &, uint64_t source, uint64_t destination, uint64_t bytes);
    // Keys are source blocks; values are destination blocks. The table keeps only owned blocks.
    Range MapBlocks(std::span<const uvec2>);
    uint32_t MapHandle(Range, uint32_t) const;
    // Span remaps a [first,end) pair by its first handle, preserving its length.
    void RebaseByBlock(mtl::Buffer &, uint64_t byte_offset, uint32_t count, Range blocks, uint32_t stride = 1u, uint32_t source_origin = 0u, uint32_t destination_origin = 0u, bool span = false);
    // Copies the records of `index`'s members in rank order, `bytes` each, to the `count` records from `destination`.
    void GatherByRank(mtl::Buffer &, uint32_t destination, uint32_t count, uint32_t bytes, MeshletIndexRef index);
    // Replaces `count` handles `stride` words apart from `byte_offset` by `first` plus their rank among `index`'s members, keeping the null sentinel.
    void RebaseByRank(mtl::Buffer &, uint64_t byte_offset, uint32_t count, MeshletIndexRef index, uint32_t first, uint32_t stride = 1u);
    // Records copies and gathers before reference remapping; the chain retains all tables.
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
    Runs<BlockRebase> BlockRebases;
    std::vector<uvec2> Blocks;
    Runs<RankRebase> RankRebases;
};
