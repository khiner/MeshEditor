#include "mesh/MeshConnectivityGpu.h"

#include "Profile.h"
#include "gpu/MeshConnectivityJob.h"
#include "gpu/MeshConnectivityPushConstants.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"

#include "state/Scene.h"

#include <cstdlib>
#include <print>

namespace {
// A submit's scratch stays under this, so a batch of large meshes splits across submits.
constexpr uint32_t ScratchWordBudget{96u << 20};

enum Domain : uint32_t { Vertices, Halfedges, Blocks, Words, WordBlocks, DomainCount };
using Batch = TiledJobBatch<MeshConnectivityJob, DomainCount>;

constexpr std::array Passes{
    TiledPass{MeshPass::ConnectivityZero, Vertices},
    TiledPass{MeshPass::ConnectivityCount, Halfedges},
    TiledPass{MeshPass::ConnectivityBlockSum, Blocks},
    TiledPass{MeshPass::ConnectivityBlockPrefix, PerJob},
    TiledPass{MeshPass::ConnectivityOffsets, Blocks},
    TiledPass{MeshPass::ConnectivityScatter, Halfedges},
    TiledPass{MeshPass::ConnectivityPair, Vertices},
    TiledPass{MeshPass::ConnectivityBits, Words},
    TiledPass{MeshPass::ConnectivityWordBlockSum, WordBlocks},
    TiledPass{MeshPass::ConnectivityWordBlockPrefix, PerJob},
    TiledPass{MeshPass::ConnectivityRanks, WordBlocks},
    // The samples run one thread per edge-first word, which the word tiles already cover.
    TiledPass{MeshPass::ConnectivitySamples, Words},
};

// Returns scratch words for bucketed halfedges, scan intermediates, and state.
uint32_t ScratchWords(uint32_t vertex_count, uint32_t halfedge_count) {
    const uint32_t words = BitWords(halfedge_count);
    return 2 * (vertex_count + 1) + halfedge_count + words + 1 +
        TileCount(vertex_count + 1, BlockElements) + TileCount(words + 1, BlockElements) + 2;
}

void SubmitChunk(state::Scene &r, std::span<const ConnectivityTarget> chunk, Batch &batch, std::vector<ConnectivityTarget> &rejected) {
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &arenas = meshes.Arenas();
    batch.Begin();
    for (const auto &target : chunk) {
        meshes.CaptureConnectivityWrite(target.StoreId);
        const auto &record = meshes.Get(target.StoreId);
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        const auto run = arenas.Connectivity.Slotted(record.Connectivity);
        const uint32_t vertex_count = record.Vertices.Count;
        const uint32_t halfedge_count = corners.Count, words = BitWords(halfedge_count);
        const uint32_t buckets = vertex_count + 1;
        const uint32_t block_count = TileCount(buckets, BlockElements), word_block_count = TileCount(words + 1, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t counts_offset = batch.AllocateScratch(ScratchWords(vertex_count, halfedge_count));
        const uint32_t cursors_offset = counts_offset + buckets;
        const uint32_t items_offset = cursors_offset + buckets;
        const uint32_t block_offset = items_offset + halfedge_count;
        const uint32_t popcount_offset = block_offset + block_count;
        const uint32_t word_block_offset = popcount_offset + words + 1;
        batch.AddJob(
            MeshConnectivityJob{
                .Corners = {corners.Slot, corners.Offset},
                .Connectivity = {run.Slot, run.Offset},
                .VertexCount = vertex_count,
                .HalfedgeCount = halfedge_count,
                .WordCount = words,
                .CountsOffset = counts_offset,
                .CursorsOffset = cursors_offset,
                .ItemsOffset = items_offset,
                .BlockOffset = block_offset,
                .BlockCount = block_count,
                .PopcountOffset = popcount_offset,
                .WordBlockOffset = word_block_offset,
                .WordBlockCount = word_block_count,
                .StateOffset = word_block_offset + word_block_count,
            },
            {TileCount(buckets, TileElements), TileCount(halfedge_count, TileElements), block_count, TileCount(words + 1, TileElements), word_block_count}
        );
    }
    batch.Submit(r.ctx().get<const mtl::Context>(), r.ctx().get<const mtl::BindlessSet>(), GetMeshPipelines(r), MeshConnectivityPushConstants{}, Passes);

    const auto scratch = batch.ScratchSpan();
    // MESHEDITOR_CONNECTIVITY_CHECK builds every mesh on the CPU too and reports the first difference.
    static const bool check = std::getenv("MESHEDITOR_CONNECTIVITY_CHECK") != nullptr;
    if (check) {
        for (uint32_t i = 0; i < chunk.size(); ++i) {
            const auto &job = batch.Jobs[i];
            if (scratch[job.StateOffset + 1] != 0) continue;
            const auto words = BitWords(job.HalfedgeCount);
            std::vector<he::HH> outgoing(job.VertexCount), opposites(job.HalfedgeCount);
            std::vector<uint32_t> bits(words), ranks(words), samples(words);
            const ConnectivityStorage host{
                .OutgoingHalfedges = outgoing,
                .Opposites = opposites,
                .EdgeFirstBits = bits,
                .EdgeFirstRanks = ranks,
                .EdgeSamples = samples,
                .Faces = {},
            };
            const auto built = BuildConnectivity(chunk[i].Data->FaceOffsets, arenas.FaceCorners.Get(meshes.Get(chunk[i].StoreId).FaceCorners), job.VertexCount, host);
            const auto gpu = meshes.GetConnectivity(chunk[i].StoreId);
            const auto report = [&](std::string_view what, uint32_t at, uint32_t got, uint32_t wanted) {
                std::println(stderr, "Connectivity: mesh {} {} {} is {} against {}", i, what, at, got, wanted);
            };
            if (scratch[job.StateOffset] != built.EdgeCount) report("edge count", 0, scratch[job.StateOffset], built.EdgeCount);
            for (uint32_t h = 0; h < job.HalfedgeCount; ++h) {
                if (*gpu.Opposites[h] == *opposites[h]) continue;
                report("opposite of", h, *gpu.Opposites[h], *opposites[h]);
                break;
            }
            for (uint32_t v = 0; v < job.VertexCount; ++v) {
                if (*gpu.OutgoingHalfedges[v] == *outgoing[v]) continue;
                report("outgoing of", v, *gpu.OutgoingHalfedges[v], *outgoing[v]);
                break;
            }
            for (uint32_t w = 0; w < words; ++w) {
                if (gpu.EdgeFirstBits[w] == bits[w] && gpu.EdgeFirstRanks[w] == ranks[w]) continue;
                report("edge word", w, gpu.EdgeFirstBits[w], bits[w]);
                break;
            }
            for (uint32_t sample = 0; sample < gpu.EdgeSamples.size(); ++sample) {
                if (gpu.EdgeSamples[sample] == samples[sample]) continue;
                report("edge sample", sample, gpu.EdgeSamples[sample], samples[sample]);
                break;
            }
        }
    }

    for (uint32_t i = 0; i < chunk.size(); ++i) {
        // A mesh with a third halfedge on an edge goes back to the store, whose build has tables for it.
        const auto state = batch.Jobs[i].StateOffset;
        if (scratch[state + 1] != 0) rejected.emplace_back(chunk[i]);
        else meshes.SetConnectivityEdgeCount(chunk[i].StoreId, scratch[state]);
    }
}
} // namespace

std::vector<ConnectivityTarget> BuildConnectivityNow(state::Scene &r, std::span<const ConnectivityTarget> targets) {
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<ConnectivityTarget> work, rejected;
    for (const auto &target : targets) {
        const uint32_t faces = target.Data->FaceCount(), corners = meshes.Get(target.StoreId).FaceCorners.Count;
        // The passes read a halfedge's face loop arithmetically, which only a triangle mesh allows.
        if (faces > 0 && corners == 3 * faces) work.emplace_back(target);
        else rejected.emplace_back(target);
    }
    if (work.empty()) return rejected;

    const profile::CpuScope scope{"ConnectivityGpu"};
    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) {
        const auto &record = meshes.Get(work[i].StoreId);
        return ScratchWords(record.Vertices.Count, record.FaceCorners.Count);
    });
    // Every chunk writes over the same buffers, so a many-mesh batch takes no fresh allocation per submit.
    Batch batch{meshes.BufferContext(), split.WidestWords, split.MostJobs};
    for (const auto chunk : split.Chunks) {
        SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), batch, rejected);
    }
    return rejected;
}
