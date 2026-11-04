#include "mesh/BoundaryLoops.h"

#include "gpu/BoundaryLoopsPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

std::vector<std::vector<std::array<uint32_t, 2>>> BoundaryLoops(state::Registry &r, uint32_t id, std::span<const uint32_t> edges, uint32_t max_sides) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Halfedge);
    if (!blocks.Count) return {};
    const auto &pipelines = GetMeshPipelines(r);
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto &scratch = chain.Scratch;
    const uint32_t halfedges = a.FaceCorners.HandleBound(record.FaceCorners), vertices = a.Vertices.HandleBound(record.Vertices);
    // The four link runs hold the links, then the doubled rank distances and successors once the labels converge, then the output pairs once the ranks do.
    const auto links = TakeScratch(chain, 4ull * halfedges).Offset;
    BoundaryLoopsPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Blocks = blocks,
        .HalfedgeBound = halfedges,
        .VertexBound = vertices,
        .ScratchSlot = scratch.Buffer.Slot,
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
        .Output = links,
    };
    const uint32_t threads = blocks.Count * MeshElementBlockSize;
    FillScratch(chain, pipelines, pc.Candidate, halfedges, 0u);
    FillScratch(chain, pipelines, links, 4u * halfedges, InvalidOffset);
    FillScratch(chain, pipelines, pc.Outgoing, vertices, 0u);
    FillScratch(chain, pipelines, pc.Outgoing + vertices, vertices, InvalidOffset);
    chain.Threads(pipelines[MeshPass::BoundaryLoopStarts], pc, threads);
    chain.Threads(pipelines[MeshPass::BoundaryLoopLink], pc, threads);
    chain.Threads(pipelines[MeshPass::BoundaryLoopFallback], pc, threads);
    chain.Threads(pipelines[MeshPass::BoundaryLoopNext], pc, threads);
    // The closed flags take the jump half the label rounds leave stale.
    pc.Closed = pc.Jumps + ((LoopRounds(halfedges) & 1u) ? 0u : halfedges);
    auto round_pc = RecordLoopRanks(chain, pipelines, pc, {MeshPass::BoundaryLoopLabel, MeshPass::BoundaryLoopRankInit, MeshPass::BoundaryLoopRank, MeshPass::BoundaryLoopCount}, halfedges, threads, pc.Lengths, pc.Flags);
    chain.Submit();
    const uint32_t count = scratch.Get({pc.Flags + halfedges, 1u})[0];
    if (!count) return {};
    round_pc.Records = TakeScratch(chain, 2ull * count).Offset;
    chain.Threads(pipelines[MeshPass::BoundaryLoopOutput], round_pc, threads);
    chain.Submit();
    const auto records = scratch.Get({round_pc.Records, 2u * count});
    std::vector<std::vector<std::array<uint32_t, 2>>> loops(count);
    for (uint32_t l = 0u; l < count; ++l) loops[l] = ReadScratch<std::array<uint32_t, 2>>(chain, pc.Output + 2u * records[2u * l], records[2u * l + 1u]);
    return loops;
}
