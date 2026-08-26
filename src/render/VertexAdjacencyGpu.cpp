#include "render/VertexAdjacencyGpu.h"

#include "Profile.h"
#include "gpu/VertexAdjacencyJob.h"
#include "gpu/VertexAdjacencyPushConstants.h"
#include "mesh/Mesh.h"
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
constexpr uint32_t ScratchWordBudget{48u << 20};

constexpr uint32_t BitWordCount(uint32_t bits) { return (bits + 31u) / 32u; }

struct AdjacencyWork {
    Mesh MeshView;
    VertexAdjacencyKind Kind;
};

// Scratch words the job takes: its per-vertex counts, its scan-block sums, and an edge job's bit words.
uint32_t ScratchWords(const AdjacencyWork &work) {
    const uint32_t counts = work.MeshView.VertexCount() + 1;
    const uint32_t blocks = TileCount(counts, BlockElements);
    const uint32_t bit_words = work.Kind == VertexAdjacencyKind::Edge ? 2 * BitWordCount(work.MeshView.HalfEdgeCount()) : 0;
    return counts + blocks + bit_words;
}

void SubmitChunk(entt::registry &r, std::span<const AdjacencyWork> chunk) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    std::vector<VertexAdjacencyJob> jobs;
    jobs.reserve(chunk.size());
    std::vector<uvec2> vertex_tiles, halfedge_tiles, block_tiles;
    uint32_t scratch_words = 0;
    for (const auto &work : chunk) {
        const auto id = work.MeshView.GetStoreId();
        const bool fan = work.Kind == VertexAdjacencyKind::Fan;
        const auto corners = meshes.GetFaceCornerRange(id);
        const auto csr = fan ? meshes.GetVertexFanAdjacencyRange(id) : meshes.GetVertexEdgeAdjacencyRange(id);
        const uint32_t vertex_count = work.MeshView.VertexCount(), halfedge_count = work.MeshView.HalfEdgeCount();
        const uint32_t counts = vertex_count + 1, block_count = TileCount(counts, BlockElements);
        const uint32_t bit_words = fan ? 0u : BitWordCount(halfedge_count);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t counts_offset = scratch_words;
        const uint32_t block_offset = counts_offset + counts;
        const uint32_t bits_offset = block_offset + block_count;
        const VertexAdjacencyJob job{
            .Corners = {corners.Slot, corners.Offset},
            .VertexCount = vertex_count,
            .HalfedgeCount = halfedge_count,
            .Kind = work.Kind,
            .CsrOffset = csr.Offset,
            .CountsOffset = counts_offset,
            .BlockOffset = block_offset,
            .BlockCount = block_count,
            .EdgeFirstBitsOffset = fan ? InvalidOffset : bits_offset,
            .EdgeFirstRanksOffset = fan ? InvalidOffset : bits_offset + bit_words,
        };
        scratch_words += counts + block_count + 2 * bit_words;
        const auto job_index = uint32_t(jobs.size());
        for (uint32_t t = 0, n = TileCount(counts, TileElements); t < n; ++t) vertex_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0, n = TileCount(halfedge_count, TileElements); t < n; ++t) halfedge_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0; t < block_count; ++t) block_tiles.emplace_back(job_index, t);
        jobs.emplace_back(job);
    }

    mtl::Buffer scratch{buffers.Ctx, uint64_t(scratch_words) * sizeof(uint32_t), SlotType::Buffer};
    const auto scratch_words_span = scratch.GetMutableSpan<uint32_t>({0, scratch_words});
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        if (chunk[i].Kind == VertexAdjacencyKind::Fan) continue;
        const auto &c = chunk[i].MeshView.GetConnectivity();
        std::ranges::copy(c.EdgeFirstBits, scratch_words_span.begin() + jobs[i].EdgeFirstBitsOffset);
        std::ranges::copy(c.EdgeFirstRanks, scratch_words_span.begin() + jobs[i].EdgeFirstRanksOffset);
    }

    std::vector<uvec2> tiles;
    tiles.reserve(vertex_tiles.size() + halfedge_tiles.size() + block_tiles.size());
    tiles.insert(tiles.end(), vertex_tiles.begin(), vertex_tiles.end());
    tiles.insert(tiles.end(), halfedge_tiles.begin(), halfedge_tiles.end());
    tiles.insert(tiles.end(), block_tiles.begin(), block_tiles.end());
    const mtl::Buffer job_buffer{buffers.Ctx, as_bytes(jobs), SlotType::Buffer};
    const mtl::Buffer tile_buffer{buffers.Ctx, as_bytes(tiles), SlotType::Buffer};

    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    const VertexAdjacencyPushConstants pc{
        .JobsSlot = job_buffer.Slot,
        .TileMapSlot = tile_buffer.Slot,
        .ScratchSlot = scratch.Slot,
        .AdjacencySlot = meshes.GetAdjacencySlot(),
    };
    const auto dispatch = [&](const mtl::ComputePipeline &pipeline, size_t groups, uint32_t first_tile) {
        encode::DispatchTiledPass(encoder, pipeline, slots, buffers, pc, groups, first_tile);
    };
    const auto first_halfedge_tile = uint32_t(vertex_tiles.size());
    const auto first_block_tile = first_halfedge_tile + uint32_t(halfedge_tiles.size());
    const auto &passes = pipelines.VertexAdjacency;
    dispatch(passes.Zero, vertex_tiles.size(), 0);
    dispatch(passes.Count, halfedge_tiles.size(), first_halfedge_tile);
    dispatch(passes.BlockSum, block_tiles.size(), first_block_tile);
    // The block prefix runs one threadgroup per job, so it reads jobs by threadgroup rather than by tile.
    dispatch(passes.BlockPrefix, jobs.size(), 0);
    dispatch(passes.Offsets, block_tiles.size(), first_block_tile);
    dispatch(passes.Scatter, halfedge_tiles.size(), first_halfedge_tile);
    dispatch(passes.Sort, vertex_tiles.size(), 0);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}
} // namespace

void BuildVertexAdjacencyNow(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    std::vector<AdjacencyWork> work;
    for (const auto entity : mesh_entities) {
        const auto mesh = TryGetMesh(r, entity);
        if (!mesh) continue;
        const auto id = mesh->GetStoreId();
        if (meshes.GetVertexFanAdjacencyRange(id).Count > 0 && BuildsFanAdjacencyOnGpu(*mesh)) {
            work.emplace_back(*mesh, VertexAdjacencyKind::Fan);
        }
        if (meshes.GetVertexEdgeAdjacencyRange(id).Count > 0 && BuildsEdgeAdjacencyOnGpu(*mesh)) {
            work.emplace_back(*mesh, VertexAdjacencyKind::Edge);
        }
    }
    if (work.empty()) return;

    const profile::CpuScope scope{"VertexAdjacencyGpu"};
    // MESHEDITOR_ADJACENCY_CHECK rebuilds every filled table on the CPU and reports the first entry that differs.
    static const bool check = std::getenv("MESHEDITOR_ADJACENCY_CHECK") != nullptr;
    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) { return ScratchWords(work[i]); });
    for (const auto chunk : split.Chunks) SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count));
    if (!check) return;
    for (const auto &item : work) {
        if (const auto mismatch = meshes.CheckVertexAdjacency(item.MeshView); !mismatch.empty()) {
            std::println(stderr, "Vertex adjacency: {}", mismatch);
        }
    }
}
