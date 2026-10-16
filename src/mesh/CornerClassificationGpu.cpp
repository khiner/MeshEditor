#include "mesh/MeshStore.h"

#include "mesh/CornerClassBatch.h"
#include "mesh/ScratchChunks.h"
#include "render/ElementWorkOps.h"

namespace {
constexpr std::array CountPasses{TiledPass{MeshPass::CornerClassificationCount, 0}, TiledPass{MeshPass::CornerClassificationPlan, 0}};
constexpr std::array WritePasses{TiledPass{MeshPass::CornerClassificationWrite, 0}};

CornerClassificationPushConstants ClassificationConstants(const MeshArenas &a) {
    return {.VertexBlocksSlot = a.Vertices.Blocks.Buffer.Slot, .EdgeSharpnessSlot = a.EdgeSharpness.Buffer.Slot, .FaceSharpnessSlot = a.FaceSharpness.Buffer.Slot};
}
} // namespace

CornerClassBatch MeshStore::EncodeCornerClassification(state::Scene &r, mtl::ComputeChain &chain, std::span<const CornerClassRequest> requests) {
    CornerClassBatch batch{.Jobs = {chain.Scratch, 2u * uint32_t(requests.size())}};
    auto &jobs = batch.Jobs;
    jobs.Begin();
    // Dirty and needed blocks are the arena blocks of a mesh's corner set.
    const auto corner_blocks = Buffers.FaceCorners.Capacity() / MeshElementBlockSize;
    for (const auto &request : requests) {
        const auto id = request.StoreId;
        const auto &record = Records.at(id);
        if (!record.TriangleCount) {
            auto &written = WriteRecord(id);
            Buffers.CornerSectors.Release(written.CornerSectors);
            Buffers.NormalSectors.Release(written.NormalSectors);
            written.Classification = uint32_t(CornerClassMode::UniformVertex);
            DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
            continue;
        }
        const bool all = request.Vertices.Storage.Slot == InvalidSlot;
        const auto vertex_count = all ? Buffers.Vertices.Count(record.Vertices) : request.VertexCount;
        if (!vertex_count) continue;
        const auto incoming = all ? Buffers.FaceCorners.Count(record.FaceCorners) : request.Incoming;
        const auto block_bound = std::min(incoming, Buffers.FaceCorners.Set(record.FaceCorners).BlockCount);
        const auto job = jobs.AddJob(
            {
                .Connectivity = GetConnectivityRef(id),
                .Vertices = all ? ElementWork{} : request.Vertices,
                .VertexCount = vertex_count,
                .FaceCount = Buffers.FaceTriangles.Count(record.FaceData),
                .CornerSectors = Buffers.CornerSectors.Ref(record.CornerSectors),
                .DirtyBlocks = AllocateElementWork(chain.Scratch, corner_blocks, block_bound),
                .NeededBlocks = AllocateElementWork(chain.Scratch, corner_blocks, block_bound),
                .State = jobs.AllocateScratch(2u),
            },
            {all ? 0u : TileCount(vertex_count, TileElements)}
        );
        if (all) Buffers.Vertices.ForEachBlock(record.Vertices, [&](uint32_t b, const auto &block) { if (block.Count) jobs.Tiles[0].emplace_back(job, b); });
        batch.Targets.push_back({id, incoming, request.Complete || all});
    }
    if (jobs.Jobs.empty()) return batch;
    std::ranges::fill(jobs.ScratchSpan().subspan(jobs.ArgumentWords), 0u);
    // The count and the plan read the same fans and write separate results.
    chain.Concurrent([&] { jobs.Encode(chain, GetMeshPipelines(r), ClassificationConstants(Buffers), CountPasses); });
    return batch;
}

void MeshStore::PlanCornerClassification(state::Scene &r, mtl::ComputeChain &chain, CornerClassBatch &batch) {
    auto &jobs = batch.Jobs;
    if (jobs.Jobs.empty()) return;
    const auto state = jobs.ScratchSpan();
    std::vector<uint8_t> writes(jobs.Jobs.size(), 0u);
    for (uint32_t i = 0u; i < jobs.Jobs.size(); ++i) {
        const auto &job = jobs.Jobs[i];
        const auto [id, incoming, complete] = batch.Targets[i];
        if (state[job.State] > incoming) throw std::logic_error("Corner classification exceeds its incoming corner bound.");
        const auto flags = state[job.State + 1u];
        CheckElementWork(chain.Scratch, job.DirtyBlocks);
        CheckElementWork(chain.Scratch, job.NeededBlocks);
        auto &record = WriteRecord(id);
        const auto uniform = !(flags & 2u) ? uint32_t(CornerClassMode::UniformFace) : !(flags & 5u) ? uint32_t(CornerClassMode::UniformVertex) :
                                                                                                      uint32_t(CornerClassMode::Mixed);
        // A local result can preserve an already-uniform mode.
        // Once a mesh is mixed, unaffected faces keep using their canonical flags and roots.
        record.Classification = complete || uniform == record.Classification ? uniform : uint32_t(CornerClassMode::Mixed);
        DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
        // Allocate mixed-sector storage on demand, initializing corners to vertex normals.
        // Uniform meshes require no sector storage.
        if (record.Classification != uint32_t(CornerClassMode::Mixed)) {
            Buffers.CornerSectors.Release(record.CornerSectors);
            Buffers.NormalSectors.Release(record.NormalSectors);
            continue;
        }
        if (!record.CornerSectors.Count) {
            if (ElementWorkEmpty(chain.Scratch, job.NeededBlocks)) continue;
            const auto blocks = Buffers.FaceCorners.HandleBound(record.FaceCorners) / MeshElementBlockSize;
            Buffers.CornerSectors.Cover(record.CornerSectors, blocks);
            Buffers.CornerSectors.Clear(record.CornerSectors, {0u, blocks});
            Buffers.NormalSectors.Cover(record.NormalSectors, blocks);
            Buffers.NormalSectors.Clear(record.NormalSectors, {0u, blocks});
        } else {
            // Only the dirty corner blocks' roots change.
            std::vector<uint32_t> written;
            ForEachWorkElement(chain.Scratch, job.DirtyBlocks, [&](uint32_t b) { written.push_back(record.CornerSectors.Offset + Buffers.FaceCorners.Blocks.Get({b, 1u})[0].Local); });
            Buffers.CornerSectors.Values.Buffer.CaptureWriteElements(written, sizeof(decltype(Buffers.CornerSectors)::Block));
        }
        writes[i] = 1u;
    }
    for (uint32_t i = 0u; i < jobs.Jobs.size(); ++i) {
        if (writes[i]) jobs.Jobs[i].CornerSectors = Buffers.CornerSectors.Ref(Records[batch.Targets[i].StoreId].CornerSectors);
    }
    std::erase_if(jobs.Tiles[0], [&](uvec2 tile) { return !writes[tile.x]; });
    if (jobs.Tiles[0].empty()) return;
    jobs.Encode(chain, GetMeshPipelines(r), ClassificationConstants(Buffers), WritePasses);
}

void MeshStore::UpdateCornerClassification(state::Scene &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    std::vector<CornerClassRequest> requests;
    requests.reserve(ids.size());
    for (const auto id : ids) requests.push_back({.StoreId = id});
    auto batch = EncodeCornerClassification(r, chain, requests);
    chain.Submit();
    PlanCornerClassification(r, chain, batch);
}
