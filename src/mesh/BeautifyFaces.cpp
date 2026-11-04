#include "mesh/BeautifyFaces.h"
#include "Profile.h"
#include "gpu/BeautifyPushConstants.h"
#include "mesh/Mesh.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

std::optional<MeshTopologyTask> BeautifyFaceTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
    const profile::CpuScope scope{"PlanBeautify"};
    if (selection.Faces.empty()) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &a = meshes.Arenas();
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const uint32_t nodes = 3u * uint32_t(selection.Faces.size()), edge_bound = a.EdgeHalfedges.HandleBound(record.EdgeData);
    // Soup edges number at most the corners, so twice the corners in slots keeps the table under half full.
    const uint32_t slots = std::bit_ceil(2u * nodes);
    const uint64_t scratch_words = uint64_t(selection.Faces.size()) + selection.Edges.size() + uint64_t(BeautifyNodeWords + 3u + BeautifyEdgeWords + 1u) * nodes + edge_bound + uint64_t(KeyTableWords) * slots + TileCount(nodes + 1u, BlockElements) + 4u;
    mtl::ComputeChain chain{meshes.BufferContext(), scratch_words};
    auto &scratch = chain.Scratch;
    BeautifyPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .VertexSlot = a.Vertices.Buffer.Slot,
        .ScratchSlot = scratch.Buffer.Slot,
        .Faces = scratch.Allocate(selection.Faces).Offset,
        .SelectedEdges = selection.Edges.empty() ? 0u : scratch.Allocate(selection.Edges).Offset,
        .SelectedEdgeCount = uint32_t(selection.Edges.size()),
        .Nodes = TakeScratch(chain, uint64_t(BeautifyNodeWords) * nodes).Offset,
        .NodeCount = nodes,
        .Owners = TakeScratch(chain, uint64_t(nodes) + 1u).Offset,
        .Users = TakeScratch(chain, nodes).Offset,
        .Second = TakeScratch(chain, nodes).Offset,
        .SoupOwned = TakeScratch(chain, edge_bound).Offset,
        .Table = TakeScratch(chain, uint64_t(KeyTableWords) * slots).Offset,
        .TableMask = slots - 1u,
        .Changed = TakeScratch(chain, 1u).Offset,
        .Rotations = TakeScratch(chain, 1u).Offset,
    };
    FillScratch(chain, pipelines, pc.Table, KeyTableWords * slots, InvalidOffset);
    FillScratch(chain, pipelines, pc.Users, nodes, 0u);
    FillScratch(chain, pipelines, pc.Second, nodes, 0u);
    FillScratch(chain, pipelines, pc.SoupOwned, edge_bound, 0u);
    chain.Threads(pipelines[MeshPass::BeautifyCorners], pc, nodes);
    chain.Threads(pipelines[MeshPass::BeautifyOwners], pc, nodes);
    chain.Threads(pipelines[MeshPass::BeautifyUsers], pc, nodes + 1u);
    ScanScratch(chain, pipelines, pc.Owners, nodes + 1u);
    chain.Submit();
    const uint32_t edges = scratch.Get({pc.Owners + nodes, 1u})[0];
    if (!edges) return {};
    pc.Edges = TakeScratch(chain, uint64_t(BeautifyEdgeWords) * edges).Offset;
    pc.EdgeCount = edges;
    pc.Flags = TakeScratch(chain, edges).Offset;
    chain.Threads(pipelines[MeshPass::BeautifyEdges], pc, nodes);
    chain.Threads(pipelines[MeshPass::BeautifyNodes], pc, nodes);
    scratch.GetMutable({pc.Rotations, 1u})[0] = 0u;
    // Off a plane each quad measures in its own plane, so the edge count caps the rounds.
    uint32_t rounds = 0u;
    for (; rounds < edges; ++rounds) {
        scratch.GetMutable({pc.Changed, 1u})[0] = 0u;
        FillScratch(chain, pipelines, pc.Table, KeyTableWords * slots, InvalidOffset);
        chain.Threads(pipelines[MeshPass::BeautifyInsertCurrent], pc, edges);
        chain.Threads(pipelines[MeshPass::BeautifyCost], pc, edges);
        chain.Threads(pipelines[MeshPass::BeautifySelect], pc, edges);
        chain.Threads(pipelines[MeshPass::BeautifyResolve], pc, edges);
        chain.Threads(pipelines[MeshPass::BeautifyFlip], pc, edges);
        chain.Submit();
        if (!scratch.Get({pc.Changed, 1u})[0]) break;
    }
    const auto rotations = scratch.Get({pc.Rotations, 1u})[0];
    profile::RecordCounter("BeautifyRotations", rotations);
    profile::RecordCounter("BeautifyRounds", rounds);
    if (!rotations) return {};
    const auto final_nodes = scratch.Get({pc.Nodes, BeautifyNodeWords * nodes});
    const auto final_edges = scratch.Get({pc.Edges, BeautifyEdgeWords * edges});
    std::vector<uint32_t> faces;
    for (uint32_t i = 0u; i < selection.Faces.size(); ++i) {
        if (final_nodes[BeautifyNodeWords * 3u * i] != InvalidOffset) faces.push_back(i);
    }
    MeshTopologyTask task{.SourceId = id, .Op = MeshTopologyOp::ReplaceFaces, .List = {0u, uint32_t(faces.size())}, .Selection = selection};
    task.List.resize(2u + faces.size());
    for (uint32_t f = 0u; f < faces.size(); ++f) {
        task.List[2u + f] = uint32_t(task.List.size());
        task.List.insert(task.List.end(), {selection.Faces[faces[f]], 1u, 3u});
        for (uint32_t i = 0u; i < 3u; ++i) {
            const uint32_t n = 3u * faces[f] + i, previous = n - n % 3u + (n + 2u) % 3u;
            task.List.insert(task.List.end(), {final_nodes[BeautifyNodeWords * n], final_edges[BeautifyEdgeWords * final_nodes[BeautifyNodeWords * previous + 2u]]});
        }
    }
    return task;
}
