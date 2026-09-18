#include "mesh/VertexWeldGpu.h"

#include "Profile.h"
#include "gpu/TiledJobPushConstants.h"
#include "gpu/VertexWeldJob.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"

#include "state/Scene.h"

#include <bit>
#include <cstring>

namespace {
// A submit's scratch stays under this, so a batch of large meshes splits across submits.
constexpr uint32_t ScratchWordBudget{96u << 20};

enum Domain : uint32_t { Table,
                         Vertices,
                         Blocks,
                         Corners,
                         DomainCount };
using Batch = TiledJobBatch<VertexWeldJob, DomainCount>;

constexpr std::array Passes{
    TiledPass{MeshPass::WeldTableInit, Table},
    TiledPass{MeshPass::WeldInsert, Vertices},
    TiledPass{MeshPass::WeldMarkReps, Vertices},
    TiledPass{MeshPass::WeldBlockSum, Blocks},
    TiledPass{MeshPass::WeldBlockPrefix, PerJob},
    TiledPass{MeshPass::WeldScan, Blocks},
    TiledPass{MeshPass::WeldEmit, Vertices},
    // The corners rewrite before the compaction moves the positions they no longer point at.
    TiledPass{MeshPass::WeldRemapCorners, Corners},
    TiledPass{MeshPass::WeldCompact, Vertices},
    TiledPass{MeshPass::WeldWriteBack, Vertices},
};

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
    uint32_t TangentWordsPerVertex{};
    uint32_t RecordWords{};
};

WeldChannels Channels(const MeshStore &meshes, const WeldTarget &target) {
    const auto &record = meshes.Get(target.StoreId);
    const uint32_t target_count = record.MorphTargetCount;
    WeldChannels c{
        .Deform = meshes.Arenas().BoneDeform.Slotted(record.BoneDeform),
        .Morph = meshes.Arenas().MorphTargets.Slotted(record.MorphTargets),
        .TargetCount = target_count,
        .TangentWordsPerVertex = target.MorphTangentDeltas->empty() ? 0u : target_count * TangentWords,
    };
    c.RecordWords = PositionWords + (c.Deform.Count > 0 ? DeformWords : 0u) + target_count * MorphWords + c.TangentWordsPerVertex;
    return c;
}

// Returns scratch words for welding `count` vertices with the selected channels.
uint32_t ScratchWords(uint32_t count, const WeldChannels &c) {
    const uint32_t marks = count + 1;
    return TableSize(count) + count + marks + TileCount(marks, BlockElements) + 2 * count +
        (c.RecordWords + c.TangentWordsPerVertex) * count;
}

void SubmitChunk(state::Scene &r, std::span<const WeldTarget> chunk, Batch &batch) {
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &arenas = meshes.Arenas();
    batch.Begin();
    for (const auto &target : chunk) {
        meshes.CaptureWeldWrite(target.StoreId);
        const auto &record = meshes.Get(target.StoreId);
        const auto vertices = arenas.Vertices.Slotted(record.Vertices);
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        const auto channels = Channels(meshes, target);
        const uint32_t count = vertices.Count;
        const uint32_t table_size = TableSize(count), marks = count + 1;
        const uint32_t block_count = TileCount(marks, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t table_offset = batch.AllocateScratch(ScratchWords(count, channels));
        const uint32_t slot_offset = table_offset + table_size;
        const uint32_t flags_offset = slot_offset + count;
        const uint32_t block_offset = flags_offset + marks;
        const uint32_t remap_offset = block_offset + block_count;
        const uint32_t reps_offset = remap_offset + count;
        const uint32_t compact_offset = reps_offset + count;
        batch.AddJob(
            VertexWeldJob{
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
            },
            {TileCount(table_size, TileElements), TileCount(marks, TileElements), block_count, TileCount(corners.Count, TileElements)}
        );
    }

    // Stage host-owned tangent deltas so the same passes compare and compact them.
    const auto scratch = batch.ScratchSpan();
    for (uint32_t i = 0; i < chunk.size(); ++i) {
        if (batch.Jobs[i].TangentOffset == InvalidOffset) continue;
        const auto &deltas = *chunk[i].MorphTangentDeltas;
        std::memcpy(scratch.data() + batch.Jobs[i].TangentOffset, deltas.data(), deltas.size() * sizeof(vec3));
    }
    batch.Submit(r.ctx().get<const mtl::Context>(), r.ctx().get<const mtl::BindlessSet>(), GetMeshPipelines(r), TiledJobPushConstants{}, Passes);

    for (uint32_t i = 0; i < chunk.size(); ++i) {
        const auto &job = batch.Jobs[i];
        const uint32_t welded = scratch[job.FlagsOffset + job.Count];
        // The compacted tangent deltas go back to the host channel a glTF export reads them from.
        if (job.TangentOffset != InvalidOffset) {
            auto &deltas = *chunk[i].MorphTangentDeltas;
            deltas.resize(size_t(job.TargetCount) * welded);
            std::memcpy(deltas.data(), scratch.data() + job.TangentOffset, deltas.size() * sizeof(vec3));
        }
        meshes.ShrinkMeshSource(chunk[i].StoreId, welded);
    }
}
} // namespace

void WeldMeshesNow(state::Scene &r, std::span<const WeldTarget> targets) {
    const profile::CpuScope scope{"WeldMeshes"};
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<WeldTarget> work;
    for (const auto &target : targets) {
        if (meshes.Get(target.StoreId).Vertices.Count == 0 || target.Data->FaceCount() == 0) continue;
        work.emplace_back(target);
    }
    if (work.empty()) return;

    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) {
        return ScratchWords(meshes.Get(work[i].StoreId).Vertices.Count, Channels(meshes, work[i]));
    });
    // Every chunk writes over the same buffers, so a many-mesh batch takes no fresh allocation per submit.
    Batch batch{meshes.BufferContext(), split.WidestWords, split.MostJobs};
    for (const auto chunk : split.Chunks) SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), batch);
}
