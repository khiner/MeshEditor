#include "mesh/MeshClone.h"

#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "numeric/uvec2.h"

#include <algorithm>

namespace {
// Records one dispatch per buffer over its jobs, each job taking `tiles_of(job)` tiles of `width` threads.
template<typename Job>
void EncodeRuns(mtl::ComputeChain &chain, const mtl::ComputePipeline &pipeline, const CloneCopies::Runs<Job> &runs, uint32_t width, auto &&tiles_of) {
    if (runs.Buffers.empty()) return;
    std::vector<Job> jobs;
    std::vector<uvec2> tiles;
    std::vector<uint32_t> first_tiles;
    for (const auto &buffer_jobs : runs.Jobs) {
        first_tiles.push_back(uint32_t(tiles.size()));
        for (const auto &job : buffer_jobs) {
            const auto index = uint32_t(jobs.size());
            jobs.push_back(job);
            for (uint32_t t = 0u, count = tiles_of(job); t < count; ++t) tiles.push_back({index, t});
        }
    }
    first_tiles.push_back(uint32_t(tiles.size()));
    mtl::Buffer job_buffer{chain.Buffers, std::as_bytes(std::span{jobs}), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    mtl::Buffer tile_buffer{chain.Buffers, std::as_bytes(std::span{tiles}), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
        encoder->setComputePipelineState(pipeline.State());
        encoder->setBuffer(*job_buffer, 0, 1);
        for (uint32_t b = 0u; b < runs.Buffers.size(); ++b) {
            const auto count = first_tiles[b + 1u] - first_tiles[b];
            if (!count) continue;
            encoder->setBuffer(**runs.Buffers[b], 0, 0);
            encoder->setBuffer(*tile_buffer, uint64_t(first_tiles[b]) * sizeof(uvec2), 2);
            encoder->dispatchThreadgroups(MTL::Size(count, 1, 1), MTL::Size(width, 1, 1));
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    });
    chain.Retain(std::move(job_buffer));
    chain.Retain(std::move(tile_buffer));
}

// Extends `run` by `next` when `next` continues it at both ends.
bool Extend(ByteCopy &run, const ByteCopy &next) {
    if (run.Source + run.Bytes != next.Source || run.Destination + run.Bytes != next.Destination) return false;
    run.Bytes += next.Bytes;
    return true;
}
bool Extend(IndexRebase &, const IndexRebase &) { return false; }
bool Extend(ReferencePairCopy &run, const ReferencePairCopy &next) {
    if (run.Source + run.Count != next.Source || run.Destination + run.Count != next.Destination ||
        run.FirstDelta != next.FirstDelta || run.SecondDelta != next.SecondDelta) return false;
    run.Count += next.Count;
    return true;
}
} // namespace

template<typename Job>
void CloneCopies::Runs<Job>::Add(mtl::Buffer &buffer, const Job &job) {
    const auto found = std::ranges::find(Buffers, &buffer);
    if (found == Buffers.end()) {
        Buffers.push_back(&buffer);
        Jobs.push_back({job});
        return;
    }
    auto &jobs = Jobs[found - Buffers.begin()];
    if (!Extend(jobs.back(), job)) jobs.push_back(job);
}

void CloneCopies::Copy(mtl::Buffer &buffer, uint64_t source, uint64_t destination, uint64_t bytes) {
    if (bytes) Copies.Add(buffer, {source, destination, bytes});
}

void CloneCopies::Rebase(mtl::Buffer &buffer, uint64_t byte_offset, uint32_t count, uint32_t delta, uint32_t stride) {
    if (count && delta) Rebases.Add(buffer, {byte_offset, count, delta, stride});
}

void CloneCopies::CopyPairs(mtl::Buffer &buffer, uint32_t source, uint32_t destination, uint32_t count, uint32_t first_delta, uint32_t second_delta) {
    if (count) Pairs.Add(buffer, {source, destination, count, first_delta, second_delta});
}

void CloneCopies::Encode(mtl::ComputeChain &chain, const MeshPipelines &pipelines) {
    EncodeRuns(chain, pipelines[MeshPass::CloneCopyByteRuns], Copies, CloneRunThreads, [](const ByteCopy &job) { return uint32_t((job.Bytes + CloneCopyTileBytes - 1u) / CloneCopyTileBytes); });
    // Rebases change only copied ranges, and pair copies write fan items no copy touches.
    EncodeRuns(chain, pipelines[MeshPass::CloneRebaseIndexRuns], Rebases, CloneRunThreads, [](const IndexRebase &job) { return (job.Count + CloneRunThreads - 1u) / CloneRunThreads; });
    EncodeRuns(chain, pipelines[MeshPass::CloneCopyReferencePairs], Pairs, ClonePairThreads, [](const ReferencePairCopy &job) { return (job.Count + ClonePairThreads - 1u) / ClonePairThreads; });
}
