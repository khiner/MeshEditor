#include "render/MeshConnectivityGpu.h"

#include "Profile.h"
#include "gpu/MeshConnectivityJob.h"
#include "gpu/MeshConnectivityPushConstants.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "render/Encoding.h"
#include "render/GpuBuffers.h"
#include "render/Pipelines.h"
#include "render/ScratchChunks.h"

#include <entt/entity/registry.hpp>

#include <cstdlib>
#include <print>

namespace {
// A submit's scratch stays under this, so a batch of large meshes splits across submits.
constexpr uint32_t ScratchWordBudget{96u << 20};

constexpr uint32_t BitWords(uint32_t bits) { return (bits + 31u) / 32u; }

// Scratch words the mesh takes: two runs over its vertices for the bucket scan, the bucketed
// halfedges, the per-word popcounts, both scans' block sums, and the two state words.
uint32_t ScratchWords(uint32_t vertex_count, uint32_t halfedge_count) {
    const uint32_t words = BitWords(halfedge_count);
    return 2 * (vertex_count + 1) + halfedge_count + words + 1 +
        TileCount(vertex_count + 1, BlockElements) + TileCount(words + 1, BlockElements) + 2;
}

void SubmitChunk(entt::registry &r, std::span<const ConnectivityTarget> chunk, mtl::Buffer &scratch, mtl::Buffer &job_buffer, mtl::Buffer &tile_buffer, std::vector<ConnectivityTarget> &rejected) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    std::vector<MeshConnectivityJob> jobs;
    jobs.reserve(chunk.size());
    std::vector<uvec2> vertex_tiles, halfedge_tiles, block_tiles, word_tiles, word_block_tiles;
    uint32_t scratch_words = 0;
    for (const auto &target : chunk) {
        const auto corners = meshes.GetFaceCornerRange(target.StoreId);
        const auto run = meshes.GetConnectivityRange(target.StoreId);
        const uint32_t vertex_count = meshes.GetVerticesRange(target.StoreId).Count;
        const uint32_t halfedge_count = corners.Count, words = BitWords(halfedge_count);
        const uint32_t buckets = vertex_count + 1;
        const uint32_t block_count = TileCount(buckets, BlockElements), word_block_count = TileCount(words + 1, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t counts_offset = scratch_words;
        const uint32_t cursors_offset = counts_offset + buckets;
        const uint32_t items_offset = cursors_offset + buckets;
        const uint32_t block_offset = items_offset + halfedge_count;
        const uint32_t popcount_offset = block_offset + block_count;
        const uint32_t word_block_offset = popcount_offset + words + 1;
        jobs.emplace_back(MeshConnectivityJob{
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
        });
        scratch_words += ScratchWords(vertex_count, halfedge_count);
        const auto job_index = uint32_t(jobs.size() - 1);
        for (uint32_t t = 0, n = TileCount(buckets, TileElements); t < n; ++t) vertex_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0, n = TileCount(halfedge_count, TileElements); t < n; ++t) halfedge_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0; t < block_count; ++t) block_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0, n = TileCount(words + 1, TileElements); t < n; ++t) word_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0; t < word_block_count; ++t) word_block_tiles.emplace_back(job_index, t);
    }

    std::vector<uvec2> tiles;
    tiles.reserve(vertex_tiles.size() + halfedge_tiles.size() + block_tiles.size() + word_tiles.size() + word_block_tiles.size());
    for (const auto *list : {&vertex_tiles, &halfedge_tiles, &block_tiles, &word_tiles, &word_block_tiles}) {
        tiles.insert(tiles.end(), list->begin(), list->end());
    }
    job_buffer.Update(as_bytes(jobs));
    tile_buffer.Update(as_bytes(tiles));

    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    const MeshConnectivityPushConstants pc{
        .JobsSlot = job_buffer.Slot,
        .TileMapSlot = tile_buffer.Slot,
        .ScratchSlot = scratch.Slot,
    };
    const auto dispatch = [&](const mtl::ComputePipeline &pipeline, size_t groups, uint32_t first_tile) {
        encode::DispatchTiledPass(encoder, pipeline, slots, buffers, pc, groups, first_tile);
    };
    const auto first_halfedge_tile = uint32_t(vertex_tiles.size());
    const auto first_block_tile = first_halfedge_tile + uint32_t(halfedge_tiles.size());
    const auto first_word_tile = first_block_tile + uint32_t(block_tiles.size());
    const auto first_word_block_tile = first_word_tile + uint32_t(word_tiles.size());
    const auto &passes = pipelines.MeshConnectivity;
    dispatch(passes.Zero, vertex_tiles.size(), 0);
    dispatch(passes.Count, halfedge_tiles.size(), first_halfedge_tile);
    dispatch(passes.BlockSum, block_tiles.size(), first_block_tile);
    // The block prefixes run one threadgroup per job, so they read jobs by threadgroup rather than by tile.
    dispatch(passes.BlockPrefix, jobs.size(), 0);
    dispatch(passes.Offsets, block_tiles.size(), first_block_tile);
    dispatch(passes.Scatter, halfedge_tiles.size(), first_halfedge_tile);
    dispatch(passes.Pair, vertex_tiles.size(), 0);
    dispatch(passes.Bits, word_tiles.size(), first_word_tile);
    dispatch(passes.WordBlockSum, word_block_tiles.size(), first_word_block_tile);
    dispatch(passes.WordBlockPrefix, jobs.size(), 0);
    dispatch(passes.Ranks, word_block_tiles.size(), first_word_block_tile);
    // The samples run one thread per edge-first word, which the word tiles already cover.
    dispatch(passes.Samples, word_tiles.size(), first_word_tile);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    const auto scratch_span = scratch.GetMutableSpan<uint32_t>({0, scratch_words});
    // MESHEDITOR_CONNECTIVITY_CHECK builds every mesh on the CPU too and reports the first difference.
    static const bool check = std::getenv("MESHEDITOR_CONNECTIVITY_CHECK") != nullptr;
    if (check) {
        for (uint32_t i = 0; i < chunk.size(); ++i) {
            const auto &job = jobs[i];
            if (scratch_span[job.StateOffset + 1] != 0) continue;
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
            const auto built = BuildConnectivity(chunk[i].Data->FaceOffsets, meshes.GetFaceCorners(chunk[i].StoreId), job.VertexCount, host);
            const auto gpu = meshes.GetConnectivity(chunk[i].StoreId);
            const auto report = [&](std::string_view what, uint32_t at, uint32_t got, uint32_t wanted) {
                std::println(stderr, "Connectivity: mesh {} {} {} is {} against {}", i, what, at, got, wanted);
            };
            if (scratch_span[job.StateOffset] != built.EdgeCount) report("edge count", 0, scratch_span[job.StateOffset], built.EdgeCount);
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
        if (scratch_span[jobs[i].StateOffset + 1] != 0) rejected.emplace_back(chunk[i]);
        else meshes.SetConnectivityEdgeCount(chunk[i].StoreId, scratch_span[jobs[i].StateOffset]);
    }
}
} // namespace

std::vector<ConnectivityTarget> BuildConnectivityNow(entt::registry &r, std::span<const ConnectivityTarget> targets) {
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<ConnectivityTarget> work, rejected;
    for (const auto &target : targets) {
        const uint32_t faces = target.Data->FaceCount(), corners = meshes.GetFaceCornerRange(target.StoreId).Count;
        // The passes read a halfedge's face loop arithmetically, which only a triangle mesh allows.
        if (faces > 0 && corners == 3 * faces) work.emplace_back(target);
        else rejected.emplace_back(target);
    }
    if (work.empty()) return rejected;

    const profile::CpuScope scope{"ConnectivityGpu"};
    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) {
        return ScratchWords(meshes.GetVerticesRange(work[i].StoreId).Count, meshes.GetFaceCornerRange(work[i].StoreId).Count);
    });

    // Every chunk writes over the same buffers, so a many-mesh batch takes no fresh allocation per submit.
    auto &ctx = r.ctx().get<GpuBuffers>().Ctx;
    mtl::Buffer scratch{ctx, uint64_t(split.WidestWords) * sizeof(uint32_t), SlotType::Buffer};
    mtl::Buffer job_buffer{ctx, uint64_t(split.MostJobs) * sizeof(MeshConnectivityJob), SlotType::Buffer};
    mtl::Buffer tile_buffer{ctx, uint64_t(split.WidestWords / 32u + split.MostJobs * 8u) * sizeof(uvec2), SlotType::Buffer};
    for (const auto chunk : split.Chunks) {
        SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), scratch, job_buffer, tile_buffer, rejected);
    }
    return rejected;
}
