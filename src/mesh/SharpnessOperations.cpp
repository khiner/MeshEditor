#include "mesh/SharpnessOperations.h"
#include "SortUnique.h"
#include "gpu/EditSharpnessJob.h"
#include "mesh/CornerClassBatch.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/NormalDeriveGpu.h"
#include "state/Registry.h"

#include <cstring>
#include <ranges>

namespace {
// The corner classification that the edits gathering at one lane step share, planned once.
struct SharedClasses {
    CornerClassBatch Batch;
    bool Planned{};
};
// A sharpness edit's work between its submits.
struct SharpnessWork {
    uint32_t StoreId;
    EditSharpnessOperation Operation;
    Element Source;
    EditSharpnessJob Pc;
    std::vector<ElementWork> Seeded;
    GeometrySelection Selection;
    MeshClosure Neighborhood{};
    SharpnessOperationChange Change{};
    std::shared_ptr<SharedClasses> Classes{};
    std::function<void(std::optional<SharpnessOperationChange>)> Then;
};
// A lane's request in the classification that the edits gathering at one lane step share.
struct ClassifyEntry {
    MeshStore::CornerClassRequest Request;
    SharpnessWork *Work;
};
// Records the classification of the work's vertices with those of the other pending lanes' edits at the same step, in one batch every gathered work shares.
void GatherClassification(state::Registry &r, mtl::ComputeChain &chain, SharpnessWork &work, const MeshStore::CornerClassRequest &request) {
    // The job's block sets take a word per corner block it reaches at most.
    chain.Gather(&GetMeshPipelines(r)[MeshPass::CornerClassificationCount], std::vector{ClassifyEntry{request, &work}}, 2u * uint64_t(request.Incoming) + 64u, [&r, &chain](std::vector<ClassifyEntry> entries) {
        const auto requests = entries | std::views::transform(&ClassifyEntry::Request) | std::ranges::to<std::vector>();
        const auto batch = std::make_shared<SharedClasses>(r.Resources.get<MeshStore>().EncodeCornerClassification(r, chain, requests));
        for (const auto &gathered : entries) gathered.Work->Classes = batch;
    });
}
// Records the derive of the edited vertices' corner classification, triangles and normals once their face neighborhood completes, then the bounds, and calls the work's `Then` once those complete.
void RecordFaceNeighborhood(state::Registry &r, mtl::ComputeChain &chain, std::shared_ptr<SharpnessWork> work, ClosureSeed vertices) {
    chain.Lane(work->StoreId, [&] {
        work->Neighborhood = EncodeVertexClosure(r, chain, work->StoreId, vertices);
        work->Neighborhood.EncodeIncidence(r, chain, work->StoreId, Element::Face);
    });
    chain.AfterSubmit([&r, &chain, work] {
        const auto id = work->StoreId;
        auto &neighborhood = work->Neighborhood;
        neighborhood.Finish(chain);
        if (!neighborhood.Counts[0]) return work->Then({});
        chain.Lane(id, [&] {
            work->Change = {EncodeFaceTriangles(r, chain, id, neighborhood.Seed(Element::Face))};
            GatherClassification(r, chain, *work, {id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Counts[1], work->Source == Element::None});
        });
        chain.AfterSubmit([&r, &chain, work] {
            const auto id = work->StoreId;
            const auto &neighborhood = work->Neighborhood;
            // The first gathered edit to continue plans the shared classification, before any of them records the normals that read it.
            if (!std::exchange(work->Classes->Planned, true)) r.Resources.get<MeshStore>().PlanCornerClassification(r, chain, work->Classes->Batch);
            chain.Lane(id, [&] {
                work->Change.Triangles.Finish(chain);
                EncodeDeriveMeshNormals(r, chain, chain.Scratch, std::array{LocalNormalWork{id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]}});
            });
            chain.AfterSubmit([&r, &chain, work] {
                auto &meshes = r.Resources.get<MeshStore>();
                const auto id = work->StoreId;
                MeshStore::SelectionUpdate update{.StoreId = id};
                for (const auto [d, domain, element] : {std::tuple{0u, MeshStore::ElementDomain::Vertex, 0u}, std::tuple{1u, MeshStore::ElementDomain::Edge, 3u}, std::tuple{2u, MeshStore::ElementDomain::Face, 2u}}) {
                    update.Blocks[d] = meshes.ArenaBlocks(id, domain, chain.Scratch, work->Neighborhood.Elements[element]);
                }
                chain.Lane(id, [&] { meshes.UpdateSelection(r, chain, std::array{update}); });
                chain.AfterSubmit([work] { work->Then(work->Change); });
            });
        });
    });
}
// Records the edit of the record's selection from the source element domain, where a vertex edit sets `vertex_edges`.
void RecordSharpnessEdit(state::Registry &r, mtl::ComputeChain &chain, uint32_t id, EditSharpnessOperation operation, Element source, bool value, float angle, const GeometrySelection *selection, std::span<const uint32_t> vertex_edges, std::function<void(std::optional<SharpnessOperationChange>)> then) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &arenas = meshes.Arenas();
    const Mesh mesh{meshes, id};
    meshes.CaptureSharpnessWrite(id, operation, *selection, vertex_edges);
    const auto &record = meshes.Get(id);
    auto work = std::make_shared<SharpnessWork>(SharpnessWork{
        .StoreId = id,
        .Operation = operation,
        .Source = source,
        .Pc = {
            .VertexSelection = source == Element::Vertex ? SeedElementWorkHandles(chain.Scratch, arenas.Vertices.HandleBound(record.Vertices), selection->Vertices) : ElementWork{},
            .FaceSharpnessSlot = arenas.FaceSharpness.Buffer.Slot,
            .EdgeSharpnessSlot = arenas.EdgeSharpness.Buffer.Slot,
            .FaceNormalsSlot = arenas.BaseFaceNormals.Buffer.Slot,
            .Connectivity = meshes.GetConnectivityRef(id),
            .EdgeCount = mesh.EdgeCount(),
            .FaceCount = mesh.FaceCount(),
            .Operation = operation,
            .Value = value ? 1u : 0u,
            .CosAngle = std::cos(angle),
        },
        .Selection = *selection,
        .Then = std::move(then),
    });
    auto &pc = work->Pc;
    std::vector<ElementWorkSeedJob> seeds;
    if (source != Element::None) {
        const auto &selected = selection->Get(source);
        pc.Selected = chain.Upload(std::as_bytes(std::span{selected}));
        pc.SelectedCount = uint32_t(selected.size());
    } else {
        pc.FaceWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.FaceTriangles, record.FaceData)).Work;
        if (operation != EditSharpnessOperation::SetAllFaces) pc.EdgeWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.EdgeHalfedges, record.EdgeData)).Work;
    }
    for (const auto &seed : seeds) work->Seeded.push_back(seed.Work);
    chain.Lane(id, [&] {
        EncodeElementMembershipWork(r, chain, seeds);
        EncodeSortElementWork(r, chain, work->Seeded);
    });
    chain.AfterSubmit([&r, &chain, &pipelines, work] {
        for (const auto seeded : work->Seeded) CheckElementWork(chain.Scratch, seeded);
        const auto &pc = work->Pc;
        const auto id = work->StoreId;
        const auto source = work->Source;
        const auto &selection = work->Selection;
        // The closures record after the sharpness writes, and the next submit commits both.
        const uint32_t count = source != Element::None             ? pc.SelectedCount :
            work->Operation == EditSharpnessOperation::SetAllFaces ? pc.FaceCount :
                                                                     std::max(pc.EdgeCount, pc.FaceCount);
        ClosureSeed vertices;
        chain.Lane(id, [&] {
            mtl::GatherPass(chain, pipelines[MeshPass::EditSharpness], pc, count);
            vertices = source == Element::Face ? EncodePrimitiveClosure(r, chain, id, ListSeed(r, chain, id, Element::Face, selection.Faces)).Seed(Element::Vertex) :
                source == Element::Edge        ? EncodeEdgeVertices(r, chain, id, ListSeed(r, chain, id, Element::Edge, selection.Edges)) :
                source == Element::None        ? EncodeAllSeed(r, chain, id, Element::Vertex) :
                                                 ListSeed(r, chain, id, Element::Vertex, selection.Vertices);
        });
        if (!vertices.Count) return work->Then({});
        if (work->Operation != EditSharpnessOperation::SetVertexEdges) return RecordFaceNeighborhood(r, chain, work, vertices);
        chain.Lane(id, [&] {
            work->Neighborhood = EncodeVertexClosure(r, chain, id, vertices);
            work->Neighborhood.EncodeIncidence(r, chain, id, Element::Edge);
        });
        chain.AfterSubmit([&r, &chain, work] {
            work->Neighborhood.Finish(chain);
            ClosureSeed edge_vertices;
            chain.Lane(work->StoreId, [&] { edge_vertices = EncodeEdgeVertices(r, chain, work->StoreId, work->Neighborhood.Seed(Element::Edge)); });
            RecordFaceNeighborhood(r, chain, work, edge_vertices);
        });
    });
}

} // namespace

void RecordSharpness(state::Registry &r, mtl::ComputeChain &chain, const SharpnessOperationTarget &target, EditSharpnessOperation operation, bool value, float angle, std::function<void(std::optional<SharpnessOperationChange>)> then) {
    if (uint32_t(operation) > uint32_t(EditSharpnessOperation::SetVertexEdges) || !std::isfinite(angle)) return then({});
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto source = operation == EditSharpnessOperation::SetSelectedFaces ? Element::Face :
        operation == EditSharpnessOperation::SetSelectedEdges                 ? Element::Edge :
        operation == EditSharpnessOperation::SetVertexEdges                   ? Element::Vertex :
                                                                                Element::None;
    const auto id = target.StoreId;
    const SelectedHandles selection{meshes, id, target.Selection};
    if (!Mesh{meshes, id}.FaceCount() || (source != Element::None && selection->Get(source).empty())) return then({});
    if (operation != EditSharpnessOperation::SetVertexEdges) return RecordSharpnessEdit(r, chain, id, operation, source, value, angle, &*selection, {}, std::move(then));
    // A vertex edit sets the edges at its vertices, which read on the chain.
    RecordVertexEdges(meshes, GetMeshPipelines(r), chain, id, selection->Vertices, [&r, &chain, id, operation, source, value, angle, selected = *selection, then = std::move(then)](Gathered<VertexEdge> edges) mutable {
        std::vector<uint32_t> vertex_edges;
        for (const auto &edge : edges.Items) vertex_edges.push_back(edge.Edge);
        SortUnique(vertex_edges);
        RecordSharpnessEdit(r, chain, id, operation, source, value, angle, &selected, vertex_edges, std::move(then));
    });
}

std::optional<SharpnessOperationChange> ExecuteGeometrySharpness(state::Registry &r, mtl::ComputeChain &chain, const SharpnessOperationTarget &target, EditSharpnessOperation operation, bool value, float angle) {
    std::optional<SharpnessOperationChange> change;
    RecordSharpness(r, chain, target, operation, value, angle, [&change](std::optional<SharpnessOperationChange> made) { change = std::move(made); });
    chain.Drain();
    return change;
}
