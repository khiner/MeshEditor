#include "render/VertexWeldGpu.h"

#include "Profile.h"
#include "gpu/VertexWeldJob.h"
#include "gpu/VertexWeldPushConstants.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "render/Encoding.h"
#include "render/GpuBuffers.h"
#include "render/Pipelines.h"
#include "render/ScratchChunks.h"

#include <entt/entity/registry.hpp>

#include <bit>
#include <cstring>

namespace {
// A submit's scratch stays under this, so a batch of large meshes splits across submits.
constexpr uint32_t ScratchWordBudget{96u << 20};

// Words each key channel takes per vertex, matching the arena element the weld reads it from.
constexpr uint32_t PositionWords{sizeof(vec3) / sizeof(uint32_t)};
constexpr uint32_t DeformWords{sizeof(BoneDeformVertex) / sizeof(uint32_t)};
constexpr uint32_t MorphWords{sizeof(MorphTargetVertex) / sizeof(uint32_t)};
constexpr uint32_t TangentWords{sizeof(vec3) / sizeof(uint32_t)};

// Probing stays short at a load factor below three quarters.
uint32_t TableSize(uint32_t count) { return std::bit_ceil(count + count / 2u + 1u); }

struct WeldChannels {
    SlottedRange Deform{}, Morph{};
    uint32_t TargetCount{};
    uint32_t TangentWordsPerVertex{}; // Zero when no target authors tangent deltas.
    uint32_t RecordWords{}; // Words one welded vertex takes in the compaction staging.
};

WeldChannels Channels(const MeshStore &meshes, const WeldTarget &target) {
    const uint32_t target_count = meshes.GetMorphTargetCount(target.StoreId);
    WeldChannels c{
        .Deform = meshes.GetBoneDeformRange(target.StoreId),
        .Morph = meshes.GetMorphTargetRange(target.StoreId),
        .TargetCount = target_count,
        .TangentWordsPerVertex = target.Prepared->MorphTangentDeltas.empty() ? 0u : target_count * TangentWords,
    };
    c.RecordWords = PositionWords + (c.Deform.Count > 0 ? DeformWords : 0u) + target_count * MorphWords + c.TangentWordsPerVertex;
    return c;
}

// Scratch words a mesh of `count` vertices takes: its hash table, per-vertex slots, marks, block sums,
// the plan the compaction reads, its staging, and the tangent deltas no arena holds.
uint32_t ScratchWords(uint32_t count, const WeldChannels &c) {
    const uint32_t marks = count + 1;
    return TableSize(count) + count + marks + TileCount(marks, BlockElements) + 2 * count +
        (c.RecordWords + c.TangentWordsPerVertex) * count;
}

struct WeldBuffers {
    mtl::Buffer Scratch, Jobs, Tiles;
};

void SubmitChunk(entt::registry &r, std::span<const WeldTarget> chunk, WeldBuffers &reused) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    std::vector<VertexWeldJob> jobs;
    jobs.reserve(chunk.size());
    std::vector<uvec2> table_tiles, vertex_tiles, block_tiles, corner_tiles;
    uint32_t scratch_words = 0;
    for (const auto &target : chunk) {
        const auto vertices = meshes.GetVerticesRange(target.StoreId);
        const auto corners = meshes.GetFaceCornerRange(target.StoreId);
        const auto channels = Channels(meshes, target);
        const uint32_t count = vertices.Count;
        const uint32_t table_size = TableSize(count), marks = count + 1;
        const uint32_t block_count = TileCount(marks, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t table_offset = scratch_words;
        const uint32_t slot_offset = table_offset + table_size;
        const uint32_t flags_offset = slot_offset + count;
        const uint32_t block_offset = flags_offset + marks;
        const uint32_t remap_offset = block_offset + block_count;
        const uint32_t reps_offset = remap_offset + count;
        const uint32_t compact_offset = reps_offset + count;
        const VertexWeldJob job{
            .Positions = {vertices.Slot, vertices.Offset},
            .Corners = {corners.Slot, corners.Offset},
            .Deform = channels.Deform.Count > 0 ? SlotOffset{channels.Deform.Slot, channels.Deform.Offset} : SlotOffset{},
            .Morph = channels.TargetCount > 0 ? SlotOffset{channels.Morph.Slot, channels.Morph.Offset} : SlotOffset{},
            .TargetCount = channels.TargetCount,
            .TangentOffset = channels.TangentWordsPerVertex > 0 ? compact_offset + channels.RecordWords * count : InvalidOffset,
            .Count = count,
            .CornerCount = corners.Count,
            .TableOffset = table_offset,
            .TableMask = table_size - 1,
            .SlotOffset = slot_offset,
            .FlagsOffset = flags_offset,
            .BlockOffset = block_offset,
            .BlockCount = block_count,
            .RemapOffset = remap_offset,
            .RepsOffset = reps_offset,
            .CompactOffset = compact_offset,
            .RecordWords = channels.RecordWords,
        };
        scratch_words += ScratchWords(count, channels);
        const auto job_index = uint32_t(jobs.size());
        for (uint32_t t = 0, n = TileCount(table_size, TileElements); t < n; ++t) table_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0, n = TileCount(marks, TileElements); t < n; ++t) vertex_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0; t < block_count; ++t) block_tiles.emplace_back(job_index, t);
        for (uint32_t t = 0, n = TileCount(corners.Count, TileElements); t < n; ++t) corner_tiles.emplace_back(job_index, t);
        jobs.emplace_back(job);
    }

    auto &scratch = reused.Scratch;
    const auto scratch_span = scratch.GetMutableSpan<uint32_t>({0, scratch_words});
    std::vector<uvec2> tiles;
    tiles.reserve(table_tiles.size() + vertex_tiles.size() + block_tiles.size() + corner_tiles.size());
    tiles.insert(tiles.end(), table_tiles.begin(), table_tiles.end());
    tiles.insert(tiles.end(), vertex_tiles.begin(), vertex_tiles.end());
    tiles.insert(tiles.end(), block_tiles.begin(), block_tiles.end());
    tiles.insert(tiles.end(), corner_tiles.begin(), corner_tiles.end());
    reused.Jobs.Update(as_bytes(jobs));
    reused.Tiles.Update(as_bytes(tiles));

    // The morph tangent deltas are the one key channel no arena holds, so they stage in the scratch,
    // where the same passes compare and compact them.
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        if (jobs[i].TangentOffset == InvalidOffset) continue;
        const auto &deltas = chunk[i].Prepared->MorphTangentDeltas;
        std::memcpy(scratch_span.data() + jobs[i].TangentOffset, deltas.data(), deltas.size() * sizeof(vec3));
    }

    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    const VertexWeldPushConstants pc{
        .JobsSlot = reused.Jobs.Slot,
        .TileMapSlot = reused.Tiles.Slot,
        .ScratchSlot = scratch.Slot,
    };
    const auto dispatch = [&](const mtl::ComputePipeline &pipeline, size_t groups, uint32_t first_tile) {
        encode::DispatchTiledPass(encoder, pipeline, slots, buffers, pc, groups, first_tile);
    };
    const auto first_vertex_tile = uint32_t(table_tiles.size());
    const auto first_block_tile = first_vertex_tile + uint32_t(vertex_tiles.size());
    const auto first_corner_tile = first_block_tile + uint32_t(block_tiles.size());
    const auto &passes = pipelines.VertexWeld;
    dispatch(passes.TableInit, table_tiles.size(), 0);
    dispatch(passes.Insert, vertex_tiles.size(), first_vertex_tile);
    dispatch(passes.MarkReps, vertex_tiles.size(), first_vertex_tile);
    dispatch(passes.BlockSum, block_tiles.size(), first_block_tile);
    // The block prefix runs one threadgroup per job, so it reads jobs by threadgroup rather than by tile.
    dispatch(passes.BlockPrefix, jobs.size(), 0);
    dispatch(passes.Scan, block_tiles.size(), first_block_tile);
    dispatch(passes.Emit, vertex_tiles.size(), first_vertex_tile);
    // The corners rewrite before the compaction moves the positions they no longer point at.
    dispatch(passes.RemapCorners, corner_tiles.size(), first_corner_tile);
    dispatch(passes.Compact, vertex_tiles.size(), first_vertex_tile);
    dispatch(passes.WriteBack, vertex_tiles.size(), first_vertex_tile);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    for (uint32_t i = 0; i < chunk.size(); ++i) {
        const auto &job = jobs[i];
        const uint32_t welded = scratch_span[job.FlagsOffset + job.Count];
        // The compacted tangent deltas go back to the host channel a glTF export reads them from.
        if (job.TangentOffset != InvalidOffset) {
            auto &deltas = chunk[i].Prepared->MorphTangentDeltas;
            deltas.resize(size_t(job.TargetCount) * welded);
            std::memcpy(deltas.data(), scratch_span.data() + job.TangentOffset, deltas.size() * sizeof(vec3));
        }
        meshes.ShrinkMeshSource(chunk[i].StoreId, welded);
    }
}

} // namespace

void WeldMeshesNow(entt::registry &r, std::span<const WeldTarget> targets) {
    const profile::CpuScope scope{"WeldMeshes"};
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<WeldTarget> work;
    for (const auto &target : targets) {
        if (meshes.GetVerticesRange(target.StoreId).Count == 0 || target.Data->FaceCount() == 0) continue;
        work.emplace_back(target);
    }
    if (work.empty()) return;

    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) {
        return ScratchWords(meshes.GetVerticesRange(work[i].StoreId).Count, Channels(meshes, work[i]));
    });

    // Every chunk writes over the same buffers, so a many-mesh batch takes no fresh allocation per submit.
    auto &ctx = r.ctx().get<GpuBuffers>().Ctx;
    WeldBuffers reused{
        .Scratch = {ctx, uint64_t(split.WidestWords) * sizeof(uint32_t), SlotType::Buffer},
        .Jobs = {ctx, uint64_t(split.MostJobs) * sizeof(VertexWeldJob), SlotType::Buffer},
        .Tiles = {ctx, uint64_t(split.WidestWords / 32u + split.MostJobs * 4u) * sizeof(uvec2), SlotType::Buffer},
    };
    for (const auto chunk : split.Chunks) SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), reused);
}
