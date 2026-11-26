#include "mesh/BeautifyFaces.h"
#include "Profile.h"
#include "gpu/BeautifyJob.h"
#include "mesh/Mesh.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

namespace {
// A Beautify plan between its submits.
struct BeautifyWork {
    uint32_t StoreId;
    GeometrySelection Selection;
    BeautifyJob Pc;
    uint32_t Nodes, Slots, Edges{}, Rounds{};
    std::function<void(std::optional<MeshTopologyTask>)> Then;
};
// Reads the rotated triangles into the task once no round changed anything or every edge has had its round.
void Finish(const mtl::ComputeChain &chain, BeautifyWork &work) {
    const auto &pc = work.Pc;
    const auto &scratch = chain.Scratch;
    const auto rotations = scratch.Get({pc.Rotations, 1u})[0];
    profile::RecordCounter("BeautifyRotations", rotations);
    profile::RecordCounter("BeautifyRounds", work.Rounds);
    if (!rotations) return work.Then({});
    const auto final_nodes = scratch.Get({pc.Nodes, BeautifyNodeWords * work.Nodes});
    const auto final_edges = scratch.Get({pc.Edges, BeautifyEdgeWords * work.Edges});
    const auto &selection = work.Selection;
    std::vector<uint32_t> faces;
    for (uint32_t i = 0u; i < selection.Faces.size(); ++i) {
        if (final_nodes[BeautifyNodeWords * 3u * i] != InvalidOffset) faces.push_back(i);
    }
    MeshTopologyTask task{.SourceId = work.StoreId, .Op = MeshTopologyOp::ReplaceFaces, .List = {0u, uint32_t(faces.size())}, .Selection = selection};
    task.List.resize(2u + faces.size());
    for (uint32_t f = 0u; f < faces.size(); ++f) {
        task.List[2u + f] = uint32_t(task.List.size());
        task.List.insert(task.List.end(), {selection.Faces[faces[f]], 1u, 3u});
        for (uint32_t i = 0u; i < 3u; ++i) {
            const uint32_t n = 3u * faces[f] + i, previous = n - n % 3u + (n + 2u) % 3u;
            task.List.insert(task.List.end(), {final_nodes[BeautifyNodeWords * n], final_edges[BeautifyEdgeWords * final_nodes[BeautifyNodeWords * previous + 2u]]});
        }
    }
    work.Then(std::move(task));
}
// Records a round of rotations, and the next once it completes having changed something, up to one round per edge.
void RecordRound(const MeshPipelines &pipelines, mtl::ComputeChain &chain, std::shared_ptr<BeautifyWork> work) {
    if (work->Rounds == work->Edges) return Finish(chain, *work);
    auto &pc = work->Pc;
    chain.Scratch.GetMutable({pc.Changed, 1u})[0] = 0u;
    chain.Lane(work->StoreId, [&] {
        FillScratch(chain, pipelines, std::array{ScratchFillRun{pc.Table, KeyTableWords * work->Slots, InvalidOffset}});
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyInsertCurrent], pc, work->Edges);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyCost], pc, work->Edges);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifySelect], pc, work->Edges);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyResolve], pc, work->Edges);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyFlip], pc, work->Edges);
    });
    chain.AfterSubmit([&pipelines, &chain, work] {
        if (!chain.Scratch.Get({work->Pc.Changed, 1u})[0]) return Finish(chain, *work);
        ++work->Rounds;
        RecordRound(pipelines, chain, work);
    });
}
} // namespace

void RecordBeautifyFaceTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, std::function<void(std::optional<MeshTopologyTask>)> then) {
    const profile::CpuScope scope{"PlanBeautify"};
    if (selection.Faces.empty()) return then({});
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &a = meshes.Arenas();
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const uint32_t nodes = 3u * uint32_t(selection.Faces.size()), edge_bound = a.EdgeHalfedges.HandleBound(record.EdgeData);
    // Soup edges number at most the corners, so twice the corners in slots keeps the table under half full.
    const uint32_t slots = std::bit_ceil(2u * nodes);
    chain.Scratch.ReserveAdditional(uint64_t(selection.Faces.size()) + selection.Edges.size() + uint64_t(BeautifyNodeWords + 3u + BeautifyEdgeWords + 1u) * nodes + edge_bound + uint64_t(KeyTableWords) * slots + TileCount(nodes + 1u, BlockElements) + 4u);
    auto &scratch = chain.Scratch;
    auto work = std::make_shared<BeautifyWork>(BeautifyWork{
        .StoreId = id,
        .Selection = selection,
        .Pc = {
            .Connectivity = meshes.GetConnectivityRef(id),
            .VertexSlot = a.Vertices.Buffer.Slot,
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
        },
        .Nodes = nodes,
        .Slots = slots,
        .Then = std::move(then),
    });
    auto &pc = work->Pc;
    chain.Lane(id, [&] {
        FillScratch(chain, pipelines, std::to_array<ScratchFillRun>({{pc.Table, KeyTableWords * slots, InvalidOffset}, {pc.Users, nodes, 0u}, {pc.Second, nodes, 0u}, {pc.SoupOwned, edge_bound, 0u}}));
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyCorners], pc, nodes);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyOwners], pc, nodes);
        mtl::GatherPass(chain, pipelines[MeshPass::BeautifyUsers], pc, nodes + 1u);
        ScanScratch(chain, pipelines, std::array{Range{pc.Owners, nodes + 1u}});
    });
    chain.AfterSubmit([&pipelines, &chain, work] {
        auto &pc = work->Pc;
        const uint32_t edges = chain.Scratch.Get({pc.Owners + work->Nodes, 1u})[0];
        if (!edges) return work->Then({});
        work->Edges = edges;
        pc.Edges = TakeScratch(chain, uint64_t(BeautifyEdgeWords) * edges).Offset;
        pc.EdgeCount = edges;
        pc.Flags = TakeScratch(chain, edges).Offset;
        chain.Lane(work->StoreId, [&] {
            mtl::GatherPass(chain, pipelines[MeshPass::BeautifyEdges], pc, work->Nodes);
            mtl::GatherPass(chain, pipelines[MeshPass::BeautifyNodes], pc, work->Nodes);
        });
        chain.Scratch.GetMutable({pc.Rotations, 1u})[0] = 0u;
        RecordRound(pipelines, chain, work);
    });
}

std::optional<MeshTopologyTask> BeautifyFaceTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
    return mtl::Drained<std::optional<MeshTopologyTask>>(r.Resources.get<const MeshStore>().BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordBeautifyFaceTask(r, chain, mesh, selection, std::move(then)); });
}
