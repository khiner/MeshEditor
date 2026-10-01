#include "mesh/MeshConnectivityGpu.h"

#include "Profile.h"
#include "mesh/ConnectivityBatch.h"
#include "mesh/VertexFanBuild.h"
#include "gpu/TiledJobPushConstants.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "mesh/ScratchChunks.h"
#include "state/Scene.h"

// Probing stays short at a load factor below three quarters. Layout arithmetic
// rejects an oversized job before any 32-bit offset or bit ceiling can wrap.
uint32_t LayoutConnectivityScratch(MeshConnectivityJob &job, uint32_t first_word) {
    const uint64_t table_size = std::bit_ceil(uint64_t(job.HalfedgeCount) + job.HalfedgeCount / 2u + 1u);
    const auto words = uint32_t((uint64_t(job.HalfedgeCount) + 31u) / 32u);
    const auto source_words = uint32_t((uint64_t(job.SourceEdgeCount) + 31u) / 32u);
    const auto scan_words = words + source_words + 2u;
    const auto blocks = TileCount(scan_words, BlockElements);
    const auto retained_words = job.SourceEdgeCount ? job.HalfedgeCount : 0u;
    const auto size = table_size + 2u * uint64_t(job.HalfedgeCount) + words + source_words + 2u * scan_words + blocks +
        retained_words + job.SourceEdgeCount + 2u;
    if (uint64_t(first_word) + size > UINT32_MAX) throw std::length_error("GPU connectivity scratch exceeds its address space.");
    job.WordCount = words;
    job.SourceWordCount = source_words;
    job.ScanWordCount = scan_words;
    job.TableOffset = first_word;
    job.TableMask = uint32_t(table_size - 1u);
    job.RepOffset = job.TableOffset + uint32_t(table_size);
    job.PartnerOffset = job.RepOffset + job.HalfedgeCount;
    job.PopcountOffset = job.PartnerOffset + job.HalfedgeCount;
    job.WordBlockOffset = job.PopcountOffset + scan_words;
    job.WordBlockCount = blocks;
    job.BitsOffset = job.WordBlockOffset + blocks;
    job.RanksOffset = job.BitsOffset + words + source_words;
    job.RetainedEdgesOffset = job.SourceEdgeCount ? job.RanksOffset + scan_words : InvalidOffset;
    job.RetiredEdgesOffset = job.RanksOffset + scan_words + retained_words;
    job.StateOffset = job.RetiredEdgesOffset + job.SourceEdgeCount;
    return uint32_t(size);
}

void AddConnectivityJob(ConnectivityBatch &batch, MeshConnectivityJob job) {
    if (job.Vertices.Storage.Slot == InvalidSlot) job.Vertices.Count = job.VertexCount;
    if (job.Halfedges.Storage.Slot == InvalidSlot) job.Halfedges.Count = job.HalfedgeCount;
    if (job.Faces.Storage.Slot == InvalidSlot) job.Faces.Count = job.FaceCount;
    if (job.SourceEdges.Storage.Slot == InvalidSlot) job.SourceEdges.Count = job.SourceEdgeCount;
    batch.AllocateScratch(LayoutConnectivityScratch(job, batch.ScratchWords));
    batch.AddJob(job, {
        TileCount(std::max({job.TableMask + 1u, job.HalfedgeCount, job.VertexCount}), TileElements),
        TileCount(job.HalfedgeCount, TileElements), job.WordBlockCount,
        TileCount(job.VertexCount, TileElements), TileCount(job.FaceCount, TileElements),
        TileCount(job.SourceEdgeCount, TileElements),
    });
}

namespace {
constexpr uint32_t ScratchWordBudget{96u << 20};

uint32_t ScratchWords(const MeshStore &meshes, uint32_t id) {
    const auto &record = meshes.Get(id);
    MeshConnectivityJob job{.HalfedgeCount = meshes.Arenas().FaceCorners.Count(record.FaceCorners)};
    return LayoutConnectivityScratch(job);
}
} // namespace

void BuildConnectivityNow(state::Scene &r, std::span<const uint32_t> store_ids) {
    if (store_ids.empty()) return;
    const profile::CpuScope scope{"ConnectivityGpu"};
    auto &meshes = r.Context.get<MeshStore>();
    const auto &arenas = meshes.Arenas();
    // One chunk per submission, so a load holds one chunk's scratch at a time.
    const auto split = ChunkByScratch(uint32_t(store_ids.size()), ScratchWordBudget, [&](uint32_t i) { return ScratchWords(meshes, store_ids[i]); });
    for (const auto range : split.Chunks) {
        const auto chunk = store_ids.subspan(range.Offset, range.Count);
        uint32_t words = 0;
        for (const auto id : chunk) words += ScratchWords(meshes, id);
        mtl::ComputeChain chain{meshes.BufferContext()};
        ConnectivityBatch batch{meshes.BufferContext(), words, range.Count};
        batch.Begin();
        std::vector<uint32_t> vertex_blocks;
        for (const auto id : chunk) {
            meshes.CaptureConnectivityWrite(id);
            const auto &record = meshes.Get(id);
            const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
            const auto blocks = RunBlocks(arenas.Vertices.First(record.Vertices), arenas.Vertices.Count(record.Vertices));
            vertex_blocks.insert(vertex_blocks.end(), blocks.begin(), blocks.end());
            AddConnectivityJob(batch, MeshConnectivityJob{
                .Corners = {corners.Slot, corners.Offset},
                .Connectivity = meshes.GetConnectivityRef(id),
                .EdgeHandles = {.First = arenas.EdgeHalfedges.First(record.EdgeData)},
                .VertexCount = arenas.Vertices.Count(record.Vertices),
                .HalfedgeCount = corners.Count,
                .FaceCount = arenas.FaceTriangles.Count(record.FaceData),
                .FaceStarts = record.ConnectivityFaceStarts ? 1u : 0u,
            });
        }
        // Full output allocations already reserve their edge capacity.
        batch.Encode(chain, GetMeshPipelines(r), TiledJobPushConstants{}, ConnectivityPasses);
        std::ranges::sort(vertex_blocks);
        EncodeVertexFans(r, chain, batch.Jobs, vertex_blocks, true);
        chain.Submit();
        const auto scratch = batch.ScratchSpan();
        for (uint32_t i = 0; i < chunk.size(); ++i) meshes.FinishConnectivity(chunk[i], scratch[batch.Jobs[i].StateOffset]);
    }
}
