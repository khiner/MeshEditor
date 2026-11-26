#include "mesh/MeshClone.h"

#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "state/Registry.h"

namespace {
// Records one dispatch per buffer over its jobs, each job taking `tiles_of(job)` tiles of CloneRunThreads threads.
template<typename Job>
void EncodeRuns(mtl::ComputeChain &chain, const mtl::ComputePipeline &pipeline, const CloneCopies::Runs<Job> &runs, auto &&tiles_of) {
    if (runs.Buffers.empty()) return;
    // The tables live on the chain's scratch, which the dispatches bind directly, so copies take no slot.
    const auto tables = runs.Jobs | std::views::transform([&](const std::vector<Job> &jobs) { return mtl::UploadJobTable(chain, std::span{jobs}, jobs | std::views::transform(tiles_of) | std::ranges::to<std::vector>()); }) | std::ranges::to<std::vector>();
    chain.Encode([&](mtl::ComputeEncoder encoder) {
        pipeline.Bind(encoder);
        auto *const scratch = *chain.Scratch.Buffer;
        for (uint32_t b = 0u; b < runs.Buffers.size(); ++b) {
            const auto &table = tables[b];
            if (!table.Groups) continue;
            encoder.Cmd->SetBuffer(**runs.Buffers[b], 0, CloneBufferIndex_Data);
            encoder.Cmd->SetBuffer(scratch, uint64_t(table.Constants.Jobs.Offset) * sizeof(uint32_t), CloneBufferIndex_Jobs);
            encoder.Cmd->SetBuffer(scratch, uint64_t(table.Constants.Tiles.Offset) * sizeof(uint32_t), CloneBufferIndex_Tiles);
            encoder->dispatchThreadgroups(MTL::Size(table.Groups, 1, 1), MTL::Size(CloneRunThreads, 1, 1));
        }
        mtl::ComputeBarrier(encoder);
    });
}

// A clone pass's job and the buffer it copies within.
template<typename Job> struct BufferJob {
    mtl::Buffer *Buffer;
    Job Run;
};
// Records the runs with those the other pending lanes gather for the pipeline at the same step, or at once outside a lane.
template<typename Job>
void GatherRuns(mtl::ComputeChain &chain, const mtl::ComputePipeline &pipeline, const CloneCopies::Runs<Job> &runs, auto tiles_of) {
    std::vector<BufferJob<Job>> entries;
    for (uint32_t b = 0u; b < runs.Buffers.size(); ++b) {
        for (const auto &job : runs.Jobs[b]) entries.push_back({runs.Buffers[b], job});
    }
    // The jobs and a first tile per job, and per buffer a job count and the alignment of its jobs.
    const auto words = entries.size() * (sizeof(Job) / sizeof(uint32_t) + 1u) + runs.Buffers.size() * alignof(Job) / sizeof(uint32_t);
    chain.Gather(&pipeline, std::move(entries), words, [&chain, &pipeline, tiles_of](std::vector<BufferJob<Job>> gathered) {
        CloneCopies::Runs<Job> merged;
        for (const auto &entry : gathered) merged.Add(*entry.Buffer, entry.Run);
        EncodeRuns(chain, pipeline, merged, tiles_of);
    });
}

bool Extend(ByteCopy &run, const ByteCopy &next) {
    if (run.Source + run.Bytes != next.Source || run.Destination + run.Bytes != next.Destination || run.Copies != next.Copies || run.Step != next.Step) return false;
    run.Bytes += next.Bytes;
    return true;
}
bool Extend(HandleOffset &run, const HandleOffset &next) {
    if (run.ByteOffset + uint64_t(run.Count) * run.Stride * 4u != next.ByteOffset || run.Stride != next.Stride || run.Shift != next.Shift || run.Copies != next.Copies || run.Step != next.Step || run.ShiftStep != next.ShiftStep) return false;
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

void CloneCopies::Copy(mtl::Buffer &buffer, const ByteCopy &copy) {
    if (copy.Bytes && copy.Copies) Copies.Add(buffer, copy);
}

void CloneCopies::Offset(mtl::Buffer &buffer, const HandleOffset &offset) {
    if (offset.Count && offset.Copies && (offset.Shift || offset.ShiftStep)) Offsets.Add(buffer, offset);
}

void CloneCopies::Encode(mtl::ComputeChain &chain, const MeshPipelines &pipelines) {
    GatherRuns(chain, pipelines[MeshPass::CloneCopyByteRuns], Copies, [](const ByteCopy &job) { return uint32_t((job.Bytes + CloneCopyTileBytes - 1u) / CloneCopyTileBytes) * job.Copies; });
    GatherRuns(chain, pipelines[MeshPass::CloneOffsetHandles], Offsets, [](const HandleOffset &job) { return (job.Count + CloneRunThreads - 1u) / CloneRunThreads * job.Copies; });
}

std::vector<uint32_t> RecordClones(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    auto &meshes = r.Resources.get<MeshStore>();
    for (const auto id : ids) meshes.PlanClone(Mesh{meshes, id});
    meshes.CommitReserves();
    CloneCopies copies;
    auto clones = meshes.CloneMeshes(copies, ids, false);
    copies.Encode(chain, GetMeshPipelines(r));
    return clones;
}
std::vector<uint32_t> CloneRecords(state::Registry &r, std::span<const uint32_t> ids) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto clones = RecordClones(r, chain, ids);
    chain.Submit();
    meshes.RebuildSelectionIndex(clones);
    return clones;
}
uint32_t CloneRecord(state::Registry &r, uint32_t id) { return CloneRecords(r, std::array{id}).front(); }
