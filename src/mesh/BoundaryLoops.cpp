#include "mesh/BoundaryLoops.h"

#include "gpu/BoundaryLoopsJob.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"

void RecordBoundaryLoops(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> edges, uint32_t max_sides, std::function<void(std::vector<std::vector<std::array<uint32_t, 2>>>)> then) {
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Halfedge);
    if (!blocks.Count) return then({});
    auto &scratch = chain.Scratch;
    const uint32_t halfedges = a.FaceCorners.HandleBound(record.FaceCorners), vertices = a.Vertices.HandleBound(record.Vertices);
    // The four link runs hold the links, then the doubled rank distances and successors once the labels converge, then the output pairs once the ranks do.
    const auto links = TakeScratch(chain, 4ull * halfedges).Offset;
    auto job = BoundaryLoopsJob{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Blocks = blocks,
        .HalfedgeBound = halfedges,
        .VertexBound = vertices,
        .Edges = edges.empty() ? 0u : scratch.Allocate(edges).Offset,
        .EdgeCount = uint32_t(edges.size()),
        .MaxSides = max_sides,
        .Candidate = TakeScratch(chain, halfedges).Offset,
        .Natural = links,
        .Claim = links + halfedges,
        .Fallback = links + 2u * halfedges,
        .Next = links + 3u * halfedges,
        .Outgoing = TakeScratch(chain, 2ull * vertices).Offset,
        .Labels = TakeScratch(chain, 2ull * halfedges).Offset,
        .Jumps = TakeScratch(chain, 2ull * halfedges).Offset,
        .Distances = links + 2u * halfedges,
        .Successors = links,
        .Lengths = TakeScratch(chain, uint64_t(halfedges) + 1u).Offset,
        .Flags = TakeScratch(chain, uint64_t(halfedges) + 1u).Offset,
        .Parity = TakeScratch(chain, 1u).Offset,
        .Output = links,
    };
    chain.Lane([&] {
        FillScratch(chain, pipelines, std::to_array<ScratchFillRun>({{job.Candidate, halfedges, 0u}, {links, 4u * halfedges, InvalidOffset}, {job.Outgoing, vertices, 0u}, {job.Outgoing + vertices, vertices, InvalidOffset}}));
        for (const auto pass : {MeshPass::BoundaryLoopStarts, MeshPass::BoundaryLoopLink, MeshPass::BoundaryLoopFallback, MeshPass::BoundaryLoopNext}) mtl::GatherPass(chain, pipelines[pass], job, job.Blocks.Count * MeshElementBlockSize);
        GatherLoopRanks(chain, pipelines, {MeshPass::BoundaryLoopLabel, MeshPass::BoundaryLoopRankInit, MeshPass::BoundaryLoopRank, MeshPass::BoundaryLoopCount}, job, job.Blocks.Count, [](const BoundaryLoopsJob &gathered) { return LoopList{gathered.HalfedgeBound, gathered.Lengths, gathered.Flags}; });
    });
    chain.AfterSubmit([&pipelines, &chain, job, then = std::move(then)] mutable {
        const uint32_t count = chain.Scratch.Get({job.Flags + job.HalfedgeBound, 1u})[0];
        if (!count) return then({});
        job.Records = TakeScratch(chain, 2ull * count).Offset;
        chain.Lane([&] { mtl::GatherPass(chain, pipelines[MeshPass::BoundaryLoopOutput], job, job.Blocks.Count * MeshElementBlockSize); });
        chain.AfterSubmit([&chain, job, count, then = std::move(then)] {
            const auto records = chain.Scratch.Get({job.Records, 2u * count});
            std::vector<std::vector<std::array<uint32_t, 2>>> loops(count);
            for (uint32_t l = 0u; l < count; ++l) loops[l] = ReadScratch<std::array<uint32_t, 2>>(chain, job.Output + 2u * records[2u * l], records[2u * l + 1u]);
            then(std::move(loops));
        });
    });
}
