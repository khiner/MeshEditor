#include "mesh/MeshConnectivityGpu.h"

#include "Profile.h"
#include "gpu/MeshConnectivityJob.h"
#include "gpu/TiledJobPushConstants.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"
#include "state/Scene.h"

#include <algorithm>
#include <bit>

namespace {
// A chunk's scratch stays under this, so a batch of large meshes splits across chunks, each over its own buffers.
constexpr uint32_t ScratchWordBudget{96u << 20};

enum Domain : uint32_t { Init,
                         Halfedges,
                         WordBlocks,
                         DomainCount };
using Batch = TiledJobBatch<MeshConnectivityJob, DomainCount>;

constexpr std::array Passes{
    TiledPass{MeshPass::ConnectivityPrev, Halfedges},
    TiledPass{MeshPass::ConnectivityInit, Init},
    TiledPass{MeshPass::ConnectivityInsert, Halfedges},
    TiledPass{MeshPass::ConnectivityResolve, Halfedges},
    TiledPass{MeshPass::ConnectivityLink, Halfedges},
    TiledPass{MeshPass::ConnectivityWordBlockSum, WordBlocks},
    TiledPass{MeshPass::ConnectivityWordBlockPrefix, PerJob},
    TiledPass{MeshPass::ConnectivityRanks, WordBlocks},
    TiledPass{MeshPass::ConnectivityEdgeTables, Halfedges},
};

// Probing stays short at a load factor below three quarters.
uint32_t TableSize(uint32_t halfedge_count) { return std::bit_ceil(halfedge_count + halfedge_count / 2u + 1u); }

// Returns scratch words for the edge table, representatives, partners, the rank scan, state, edge-first bits and ranks, and staged predecessors.
uint32_t ScratchWords(uint32_t halfedge_count, bool face_starts) {
    const uint32_t words = BitWords(halfedge_count);
    return TableSize(halfedge_count) + 2 * halfedge_count + words + 1 + TileCount(words + 1, BlockElements) + 1 +
        2 * words + (face_starts ? halfedge_count : 0u);
}

void EncodeChunk(state::Scene &r, std::span<const uint32_t> chunk, Batch &batch, MTL::ComputeCommandEncoder *encoder) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto &arenas = meshes.Arenas();
    batch.Begin();
    for (const auto id : chunk) {
        meshes.CaptureConnectivityWrite(id);
        const auto &record = meshes.Get(id);
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        const auto run = arenas.Connectivity.Slotted(record.Connectivity);
        const uint32_t vertex_count = record.Vertices.Count;
        const uint32_t halfedge_count = corners.Count, words = BitWords(halfedge_count);
        const uint32_t table_size = TableSize(halfedge_count), word_block_count = TileCount(words + 1, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t table_offset = batch.AllocateScratch(ScratchWords(halfedge_count, record.ConnectivityFaceStarts));
        const uint32_t rep_offset = table_offset + table_size;
        const uint32_t partner_offset = rep_offset + halfedge_count;
        const uint32_t popcount_offset = partner_offset + halfedge_count;
        const uint32_t word_block_offset = popcount_offset + words + 1;
        const uint32_t state_offset = word_block_offset + word_block_count;
        const uint32_t bits_offset = state_offset + 1;
        batch.AddJob(
            MeshConnectivityJob{
                .Corners = {corners.Slot, corners.Offset},
                .Connectivity = {run.Slot, run.Offset},
                .VertexCount = vertex_count,
                .HalfedgeCount = halfedge_count,
                .FaceCount = record.ConnectivityFaces,
                .FaceStarts = record.ConnectivityFaceStarts ? 1u : 0u,
                .WordCount = words,
                .TableOffset = table_offset,
                .TableMask = table_size - 1,
                .RepOffset = rep_offset,
                .PartnerOffset = partner_offset,
                .PopcountOffset = popcount_offset,
                .WordBlockOffset = word_block_offset,
                .WordBlockCount = word_block_count,
                .BitsOffset = bits_offset,
                .RanksOffset = bits_offset + words,
                .PrevOffset = record.ConnectivityFaceStarts ? bits_offset + 2 * words : InvalidOffset,
                .StateOffset = state_offset,
            },
            {TileCount(std::max({table_size, halfedge_count, vertex_count}), TileElements), TileCount(halfedge_count, TileElements), word_block_count}
        );
    }
    batch.Encode(r.Context.get<const mtl::BindlessSet>(), GetMeshPipelines(r), TiledJobPushConstants{}, Passes, encoder);
}

uint32_t ScratchWords(const MeshStore &meshes, uint32_t id) {
    const auto &record = meshes.Get(id);
    return ScratchWords(record.FaceCorners.Count, record.ConnectivityFaceStarts);
}

ScratchChunks Split(const MeshStore &meshes, std::span<const uint32_t> store_ids) {
    return ChunkByScratch(uint32_t(store_ids.size()), ScratchWordBudget, [&](uint32_t i) { return ScratchWords(meshes, store_ids[i]); });
}
} // namespace

// One batch per scratch chunk, each over its own scratch, job, and tile buffers so every chunk sits in one command buffer.
// Ids run in encode order, and each batch's jobs take the next run of them.
struct PendingConnectivity::Batches {
    std::vector<Batch> Chunks;
    std::vector<uint32_t> Ids;
};

PendingConnectivity::PendingConnectivity() : Chunks{std::make_unique<Batches>()} {}
PendingConnectivity::PendingConnectivity(PendingConnectivity &&) noexcept = default;
PendingConnectivity::~PendingConnectivity() = default;

PendingConnectivity EncodeConnectivity(state::Scene &r, std::span<const uint32_t> store_ids, MTL::ComputeCommandEncoder *encoder) {
    PendingConnectivity pending;
    if (store_ids.empty()) return pending;
    auto &meshes = r.Context.get<MeshStore>();
    const auto split = Split(meshes, store_ids);
    auto &batches = *pending.Chunks;
    batches.Ids.assign(store_ids.begin(), store_ids.end());
    batches.Chunks.reserve(split.Chunks.size());
    for (const auto chunk : split.Chunks) {
        uint32_t words = 0;
        for (uint32_t i = chunk.Offset; i < chunk.Offset + chunk.Count; ++i) words += ScratchWords(meshes, store_ids[i]);
        EncodeChunk(r, store_ids.subspan(chunk.Offset, chunk.Count), batches.Chunks.emplace_back(meshes.BufferContext(), words, chunk.Count), encoder);
    }
    return pending;
}

void FinishConnectivity(state::Scene &r, PendingConnectivity &pending) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto &batches = *pending.Chunks;
    uint32_t next = 0;
    for (const auto &batch : batches.Chunks) {
        const auto scratch = batch.ScratchSpan();
        for (const auto &job : batch.Jobs) meshes.FinishConnectivity(batches.Ids[next++], scratch[job.StateOffset]);
    }
}

void BuildConnectivityNow(state::Scene &r, std::span<const uint32_t> store_ids) {
    if (store_ids.empty()) return;
    const profile::CpuScope scope{"ConnectivityGpu"};
    const auto &ctx = r.Context.get<const mtl::Context>();
    // One chunk per command buffer, so a load holds one chunk's scratch at a time.
    for (const auto chunk : Split(r.Context.get<const MeshStore>(), store_ids).Chunks) {
        auto *command_buffer = ctx.Queue->commandBuffer();
        auto *encoder = command_buffer->computeCommandEncoder();
        auto pending = EncodeConnectivity(r, store_ids.subspan(chunk.Offset, chunk.Count), encoder);
        encoder->endEncoding();
        // The chunk's buffers are allocated during encoding, so residency commits after it.
        ctx.CommitResidency();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        FinishConnectivity(r, pending);
    }
}
