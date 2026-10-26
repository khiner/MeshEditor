#pragma once

#include "gpu/MeshCloneJob.h"

#include <span>
#include <vector>

struct MeshPipelines;
namespace mtl {
struct Buffer;
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

// A clone batch copies owned byte runs, then shifts the references a join moves.
// Each buffer's runs of one kind take one dispatch, so a batch costs a fixed number of dispatches per buffer at any clone count.
// Contiguous copies and compatible shifts coalesce into runs.
struct CloneCopies {
    void Copy(mtl::Buffer &, uint64_t source, uint64_t destination, uint64_t bytes);
    // Adds `shift` to `count` handles `stride` words apart from `byte_offset`, keeping the null sentinel.
    void Offset(mtl::Buffer &, uint64_t byte_offset, uint32_t count, uint32_t shift, uint32_t stride = 1u);
    // Records copies before reference shifts, and the chain retains their tables.
    void Encode(mtl::ComputeChain &, const MeshPipelines &);

    // Each buffer's jobs of one kind.
    template<typename Job> struct Runs {
        std::vector<mtl::Buffer *> Buffers;
        std::vector<std::vector<Job>> Jobs;
        void Add(mtl::Buffer &, const Job &);
    };

private:
    Runs<ByteCopy> Copies;
    Runs<HandleOffset> Offsets;
};

// Copies the records' canonical data into new records without render data, in list order.
std::vector<uint32_t> CloneRecords(state::Registry &, std::span<const uint32_t> ids);
uint32_t CloneRecord(state::Registry &, uint32_t id);
