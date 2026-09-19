#include "mesh/VertexAdjacencyGpu.h"

#include "Profile.h"
#include "gpu/VertexAdjacencyJob.h"
#include "gpu/VertexAdjacencyPushConstants.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TiledJobBatch.h"

#include "state/Scene.h"

namespace {
// A submit's scratch stays under this, so a batch of large meshes splits across submits.
constexpr uint32_t ScratchWordBudget{48u << 20};

enum Domain : uint32_t { Vertices,
                         Halfedges,
                         Blocks,
                         DomainCount };
using Batch = TiledJobBatch<VertexAdjacencyJob, DomainCount>;

constexpr std::array Passes{
    TiledPass{MeshPass::AdjacencyZero, Vertices},
    TiledPass{MeshPass::AdjacencyCount, Halfedges},
    TiledPass{MeshPass::AdjacencyBlockSum, Blocks},
    TiledPass{MeshPass::AdjacencyBlockPrefix, PerJob},
    TiledPass{MeshPass::AdjacencyOffsets, Blocks},
    TiledPass{MeshPass::AdjacencyScatter, Halfedges},
    TiledPass{MeshPass::AdjacencySort, Vertices},
};

struct AdjacencyWork {
    Mesh MeshView;
    VertexAdjacencyKind Kind;
};

// Scratch words the job takes: its per-vertex counts and its scan-block sums.
uint32_t ScratchWords(const AdjacencyWork &work) {
    const uint32_t counts = work.MeshView.VertexCount() + 1;
    const uint32_t blocks = TileCount(counts, BlockElements);
    return counts + blocks;
}

void SubmitChunk(state::Scene &r, std::span<const AdjacencyWork> chunk, Batch &batch) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &arenas = meshes.Arenas();
    batch.Begin();
    for (const auto &work : chunk) {
        const auto id = work.MeshView.GetStoreId();
        const bool fan = work.Kind == VertexAdjacencyKind::Fan;
        const auto &derived = meshes.GetDerived(id);
        const auto &record = meshes.Get(id);
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        const auto run = arenas.Connectivity.Slotted(record.Connectivity);
        const auto csr = fan ? derived.VertexFanAdjacency : derived.VertexEdgeAdjacency;
        const uint32_t vertex_count = work.MeshView.VertexCount(), halfedge_count = work.MeshView.HalfEdgeCount();
        const uint32_t counts = vertex_count + 1, block_count = TileCount(counts, BlockElements);
        // The scratch runs follow the order ScratchWords sizes them in.
        const uint32_t counts_offset = batch.AllocateScratch(ScratchWords(work));
        const uint32_t block_offset = counts_offset + counts;
        batch.AddJob(
            VertexAdjacencyJob{
                .Corners = {corners.Slot, corners.Offset},
                .Connectivity = {run.Slot, run.Offset},
                .VertexCount = vertex_count,
                .HalfedgeCount = halfedge_count,
                .FaceCount = record.ConnectivityFaces,
                .FaceStarts = record.ConnectivityFaceStarts ? 1u : 0u,
                .Kind = work.Kind,
                .CsrOffset = csr.Offset,
                .CountsOffset = counts_offset,
                .BlockOffset = block_offset,
                .BlockCount = block_count,
            },
            {TileCount(counts, TileElements), TileCount(halfedge_count, TileElements), block_count}
        );
    }

    batch.Submit(
        r.Context.get<const mtl::Context>(), r.Context.get<const mtl::BindlessSet>(), GetMeshPipelines(r),
        VertexAdjacencyPushConstants{.AdjacencySlot = meshes.Slots().Adjacency}, Passes
    );
}
} // namespace

void BuildVertexAdjacencyNow(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<AdjacencyWork> work;
    for (const auto entity : mesh_entities) {
        const auto mesh = TryGetMesh(r, entity);
        if (!mesh) continue;
        const auto &derived = meshes.GetDerived(mesh->GetStoreId());
        if (derived.VertexFanAdjacency.Count > 0 && BuildsAdjacencyOnGpu(*mesh)) work.emplace_back(*mesh, VertexAdjacencyKind::Fan);
        if (derived.VertexEdgeAdjacency.Count > 0 && BuildsAdjacencyOnGpu(*mesh)) work.emplace_back(*mesh, VertexAdjacencyKind::Edge);
    }
    if (work.empty()) return;

    const profile::CpuScope scope{"VertexAdjacencyGpu"};
    const auto split = ChunkByScratch(uint32_t(work.size()), ScratchWordBudget, [&](uint32_t i) { return ScratchWords(work[i]); });
    // Every chunk writes over the same buffers, so a many-mesh batch takes no fresh allocation per submit.
    Batch batch{meshes.BufferContext(), split.WidestWords, split.MostJobs};
    for (const auto chunk : split.Chunks) SubmitChunk(r, std::span{work}.subspan(chunk.Offset, chunk.Count), batch);
}
