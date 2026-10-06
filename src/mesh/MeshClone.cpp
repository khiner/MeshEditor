#include "mesh/MeshClone.h"

#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "numeric/uvec2.h"

#include <algorithm>
#include <bit>

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
        encoder->setBuffer(*job_buffer, 0, CloneBufferIndex_Jobs);
        for (uint32_t b = 0u; b < runs.Buffers.size(); ++b) {
            const auto count = first_tiles[b + 1u] - first_tiles[b];
            if (!count) continue;
            encoder->setBuffer(**runs.Buffers[b], 0, CloneBufferIndex_Data);
            encoder->setBuffer(*tile_buffer, uint64_t(first_tiles[b]) * sizeof(uvec2), CloneBufferIndex_Tiles);
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
bool Extend(RankGather &, const RankGather &) { return false; }
bool Extend(BlockRebase &run, const BlockRebase &next) {
    if (run.ByteOffset + uint64_t(run.Count) * run.Stride * 4u != next.ByteOffset || run.Stride != next.Stride ||
        run.MapOffset != next.MapOffset || run.MapCapacity != next.MapCapacity || run.SourceOrigin != next.SourceOrigin || run.DestinationOrigin != next.DestinationOrigin || run.Span != next.Span) return false;
    run.Count += next.Count;
    return true;
}
bool Extend(RankRebase &run, const RankRebase &next) {
    if (run.ByteOffset + uint64_t(run.Count) * run.Stride * sizeof(uint32_t) != next.ByteOffset || run.Stride != next.Stride || run.First != next.First ||
        run.Index.NodesSlot != next.Index.NodesSlot || run.Index.LeavesSlot != next.Index.LeavesSlot || run.Index.Root != next.Index.Root) return false;
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

void CloneCopies::GatherByRank(mtl::Buffer &buffer, uint32_t destination, uint32_t count, uint32_t bytes, MeshletIndexRef index) {
    if (count) Gathers.Add(buffer, {destination, count, bytes, index});
}

Range CloneCopies::MapBlocks(std::span<const uvec2> pairs) {
    const Range range{uint32_t(Blocks.size()), std::bit_ceil(std::max(2u, uint32_t(pairs.size()) * 2u))};
    Blocks.resize(Blocks.size() + range.Count);
    for (const auto pair : pairs) {
        uint32_t slot = WorkHash(pair.x, range.Count);
        while (Blocks[range.Offset + slot].x) slot = (slot + 1u) & (range.Count - 1u);
        Blocks[range.Offset + slot] = {pair.x + 1u, pair.y};
    }
    return range;
}

uint32_t CloneCopies::MapHandle(Range range, uint32_t handle) const {
    if (handle == InvalidOffset || !range.Count) return InvalidOffset;
    const auto block = handle / MeshElementBlockSize;
    uint32_t slot = WorkHash(block, range.Count);
    while (const auto key = Blocks[range.Offset + slot].x) {
        if (key == block + 1u) return Blocks[range.Offset + slot].y * MeshElementBlockSize + handle % MeshElementBlockSize;
        slot = (slot + 1u) & (range.Count - 1u);
    }
    return InvalidOffset;
}

void CloneCopies::RebaseByBlock(mtl::Buffer &buffer, uint64_t byte_offset, uint32_t count, Range blocks, uint32_t stride, uint32_t source_origin, uint32_t destination_origin, bool span) {
    if (count) BlockRebases.Add(buffer, {byte_offset, count, stride, blocks.Offset, blocks.Count, source_origin, destination_origin, uint32_t(span)});
}

void CloneCopies::RebaseByRank(mtl::Buffer &buffer, uint64_t byte_offset, uint32_t count, MeshletIndexRef index, uint32_t first, uint32_t stride) {
    if (count) RankRebases.Add(buffer, {byte_offset, count, stride, first, index});
}

void CloneCopies::Encode(mtl::ComputeChain &chain, const MeshPipelines &pipelines) {
    EncodeRuns(chain, pipelines[MeshPass::CloneCopyByteRuns], Copies, CloneRunThreads, [](const ByteCopy &job) { return uint32_t((job.Bytes + CloneCopyTileBytes - 1u) / CloneCopyTileBytes); });
    EncodeRuns(chain, pipelines[MeshPass::CloneGatherByRank], Gathers, CloneRunThreads, [](const RankGather &job) { return (job.Count + CloneRunThreads - 1u) / CloneRunThreads; });
    mtl::Buffer maps{chain.Buffers, std::as_bytes(std::span{Blocks}), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) { encoder->setBuffer(*maps, 0, CloneBufferIndex_Maps); });
    EncodeRuns(chain, pipelines[MeshPass::CloneRebaseByBlock], BlockRebases, CloneRunThreads, [](const BlockRebase &job) { return (job.Count + CloneRunThreads - 1u) / CloneRunThreads; });
    chain.Retain(std::move(maps));
    EncodeRuns(chain, pipelines[MeshPass::CloneRebaseByRank], RankRebases, CloneRunThreads, [](const RankRebase &job) { return (job.Count + CloneRunThreads - 1u) / CloneRunThreads; });
}
