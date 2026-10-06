#include "mesh/VertexFanBuild.h"
#include "Profile.h"
#include "gpu/MeshConnectivityJob.h"
#include "gpu/VertexFanBuildJob.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "mesh/TiledJobBatch.h"
#include "state/Scene.h"

#include <memory>

namespace {
using FanBatch = TiledJobBatch<VertexFanBuildJob, 3>;

// Frees the writable vertices' former runs and each run's unused tail, which the completed rebuild reports in its scratch.
void ReleaseFormerFans(MeshStore &meshes, const FanBatch &batch) {
    const profile::CpuScope scope{"VertexFanRelease"};
    const auto scratch = batch.ScratchSpan();
    std::vector<uvec2> released;
    for (const auto &job : batch.Jobs) {
        const auto used = scratch[job.TileData + (job.VertexCount + 255u) / 256u];
        if (used > job.HalfedgeCount) throw std::logic_error("Vertex fan counts exceed supplied incidence.");
        if (used < job.HalfedgeCount) released.push_back({job.FirstItem + used, job.HalfedgeCount - used});
        if (job.Fresh) continue;
        // Only a vertex that owned a run before the rebuild contributes one to free.
        for (uint32_t i = 0u; i < job.VertexCount; ++i) {
            const uvec2 former{scratch[job.Metadata + 4u * i + 2u], scratch[job.Metadata + 4u * i + 3u]};
            if (former.y) released.push_back(former);
        }
    }
    std::ranges::sort(released, {}, [](uvec2 run) { return run.x; });
    meshes.VertexFans().Release(released);
}
} // namespace

void EncodeVertexFans(state::Scene &r, mtl::ComputeChain &chain, std::span<const MeshConnectivityJob> sources, std::span<const uint32_t> vertex_blocks, bool fresh) {
    // A source's scratch holds its vertex metadata, corner keys, order and temporaries, radix histograms, totals and tile data.
    const auto scratch_words = [&](const MeshConnectivityJob &source) {
        const uint64_t h = source.HalfedgeCount, v = source.VertexCount;
        return (fresh ? 1u : 4u) * v + (fresh ? 3u : 4u) * h + 16u * ((h + 255u) / 256u) + 17u + (v + 255u) / 256u;
    };
    uint64_t total_words = 0u;
    for (const auto &source : sources)
        if (source.VertexCount) total_words += scratch_words(source);
    if (!total_words) return;
    if (total_words > UINT32_MAX) throw std::length_error("Vertex fan scratch exceeds its address space.");
    const auto batch = std::make_shared<FanBatch>(chain.Scratch, uint32_t(total_words));
    auto &meshes = r.Context.get<MeshStore>();
    auto &fans = meshes.VertexFans();
    std::vector<Range> runs;
    batch->Begin();
    for (const auto &source : sources) {
        if (!source.VertexCount) continue;
        VertexFanBuildJob job{.Vertices = source.Vertices, .Halfedges = source.Halfedges, .Corners = source.Corners, .Roots = source.Connectivity.VertexCorners, .ItemsSlot = fans.Items.Buffer.Slot, .FaceOwnersSlot = source.Connectivity.HalfedgeFaces.Slot, .FaceCount = source.FaceCount, .VertexCount = source.VertexCount, .HalfedgeCount = source.HalfedgeCount, .Fresh = fresh ? 1u : 0u};
        if (job.Vertices.Storage.Slot == InvalidSlot) job.Vertices.Count = job.VertexCount;
        if (job.Halfedges.Storage.Slot == InvalidSlot) job.Halfedges.Count = job.HalfedgeCount;
        // Truncating InvalidOffset to this many radix bits still leaves it
        // above every valid vertex rank, including power-of-two domains.
        job.VertexKeyPasses = std::max(1u, (std::bit_width(job.VertexCount) + 3u) / 4u);
        const uint64_t h = job.HalfedgeCount, v = job.VertexCount, blocks = (h + 255u) / 256u;
        if (fresh && (job.Vertices.Storage.Slot != InvalidSlot || job.Halfedges.Storage.Slot != InvalidSlot || job.Roots.Offset % 256u)) {
            throw std::invalid_argument("Fresh vertex fans require aligned dense vertices and corners.");
        }
        job.Metadata = batch->AllocateScratch(uint32_t(scratch_words(source)));
        job.Keys = job.Metadata + uint32_t((fresh ? 1u : 4u) * v);
        job.Order = job.Keys + uint32_t((fresh ? 1u : 2u) * h);
        job.Temporary = job.Order + uint32_t(h);
        job.Histogram = job.Temporary + uint32_t(h);
        job.Totals = job.Histogram + uint32_t(16u * blocks);
        job.TileData = job.Totals + 16u;
        // Every corner joins at most one fan, so the supplied incidence bounds the run.
        if (h) {
            job.FirstItem = fans.Items.Allocate(uint32_t(h)).Offset;
            runs.push_back({job.FirstItem, uint32_t(h)});
        }
        batch->AddJob(job, {uint32_t((std::max(v, h) + 255u) / 256u), uint32_t(blocks), h ? 16u : 0u});
    }
    fans.Items.Buffer.CaptureWriteRanges(runs, sizeof(uvec2));
    PageFootprint roots;
    roots.Add(meshes.Arenas().VertexCorners.Buffer, vertex_blocks, BlockBytes<uvec2>);
    roots.CaptureWrites();
    std::vector<TiledPass> passes{{MeshPass::VertexFanInit, 0u}, {MeshPass::VertexFanKeys, 1u}, {MeshPass::VertexFanTileScan, 0u}, {MeshPass::VertexFanTilePrefix, PerJob}};
    // Dense runs and finished sparse work both enumerate corners in canonical
    // order. A stable vertex-rank sort preserves that order inside each fan.
    const uint32_t key_passes = std::ranges::max(batch->Jobs, {}, &VertexFanBuildJob::VertexKeyPasses).VertexKeyPasses;
    for (uint32_t pass = 0u; pass < key_passes; ++pass) {
        passes.push_back({MeshPass::VertexFanHistogram, 1u, pass});
        passes.push_back({MeshPass::VertexFanPrefix, 2u, pass});
        passes.push_back({MeshPass::VertexFanScatter, 1u, pass});
    }
    passes.push_back({MeshPass::VertexFanEmit, 0u});
    batch->Encode(chain, GetMeshPipelines(r), VertexFanBuildPushConstants{}, passes);
    chain.AfterSubmit([&meshes, batch] { ReleaseFormerFans(meshes, *batch); });
}
