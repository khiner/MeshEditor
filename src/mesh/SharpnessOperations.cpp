#include "mesh/SharpnessOperations.h"
#include "SortUnique.h"
#include "gpu/EditSharpnessPushConstants.h"
#include "mesh/CornerClassBatch.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/NormalDeriveGpu.h"
#include "state/Registry.h"

std::optional<SharpnessOperationChange> ExecuteGeometrySharpness(state::Registry &r, mtl::ComputeChain &chain, const SharpnessOperationTarget &target, EditSharpnessOperation operation, bool value, float angle) {
    if (uint32_t(operation) > uint32_t(EditSharpnessOperation::SetVertexEdges) || !std::isfinite(angle)) return {};
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto source = operation == EditSharpnessOperation::SetSelectedFaces ? Element::Face :
        operation == EditSharpnessOperation::SetSelectedEdges                 ? Element::Edge :
        operation == EditSharpnessOperation::SetVertexEdges                   ? Element::Vertex :
                                                                                Element::None;
    const auto id = target.StoreId;
    const SelectedHandles selection{meshes, id, target.Selection};
    const auto &arenas = meshes.Arenas();
    const Mesh mesh{meshes, id};
    if (!mesh.FaceCount() || (source != Element::None && selection->Get(source).empty())) return {};
    std::vector<uint32_t> vertex_edges;
    if (operation == EditSharpnessOperation::SetVertexEdges) {
        for (const auto &edge : VertexEdges(meshes, pipelines, id, selection->Vertices).Items) vertex_edges.push_back(edge.Edge);
        SortUnique(vertex_edges);
    }
    meshes.CaptureSharpnessWrite(id, operation, *selection, vertex_edges);
    const auto &record = meshes.Get(id);
    EditSharpnessPushConstants pc{
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
    };
    std::vector<ElementWorkSeedJob> seeds;
    if (source != Element::None) {
        const auto &selected = selection->Get(source);
        pc.Selected = chain.Upload(std::as_bytes(std::span{selected}));
        pc.SelectedCount = uint32_t(selected.size());
    } else {
        pc.FaceWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.FaceTriangles, record.FaceData)).Work;
        if (operation != EditSharpnessOperation::SetAllFaces) pc.EdgeWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.EdgeHalfedges, record.EdgeData)).Work;
    }
    std::vector<ElementWork> seeded;
    seeded.reserve(seeds.size());
    for (const auto &seed : seeds) seeded.push_back(seed.Work);
    EncodeElementMembershipWork(r, chain, seeds);
    EncodeSortElementWork(r, chain, seeded);
    chain.Submit();
    for (const auto work : seeded) CheckElementWork(chain.Scratch, work);
    // The closures record after the sharpness writes, and the next submit commits both.
    const uint32_t count = source != Element::None       ? pc.SelectedCount :
        operation == EditSharpnessOperation::SetAllFaces ? pc.FaceCount :
                                                           std::max(pc.EdgeCount, pc.FaceCount);
    chain.Groups(pipelines[MeshPass::EditSharpness], pc, TileCount(count, TileElements));
    auto vertices = source == Element::Face ? EncodePrimitiveClosure(r, chain, id, ListSeed(r, chain, id, Element::Face, selection->Faces)).Seed(Element::Vertex) :
        source == Element::Edge             ? EncodeEdgeVertices(r, chain, id, ListSeed(r, chain, id, Element::Edge, selection->Edges)) :
        source == Element::None             ? EncodeAllSeed(r, chain, id, Element::Vertex) :
                                              ListSeed(r, chain, id, Element::Vertex, selection->Vertices);
    if (!vertices.Count) {
        chain.Submit();
        return {};
    }
    MeshClosure neighborhood;
    if (operation == EditSharpnessOperation::SetVertexEdges) {
        neighborhood = EncodeVertexClosure(r, chain, id, vertices);
        neighborhood.EncodeIncidence(r, chain, id, Element::Edge);
        chain.Submit();
        neighborhood.Finish(chain);
        vertices = EncodeEdgeVertices(r, chain, id, neighborhood.Seed(Element::Edge));
    }
    neighborhood = EncodeVertexClosure(r, chain, id, vertices);
    neighborhood.EncodeIncidence(r, chain, id, Element::Face);
    chain.Submit();
    neighborhood.Finish(chain);
    if (!neighborhood.Counts[0]) return {};
    SharpnessOperationChange change{EncodeFaceTriangles(r, chain, id, neighborhood.Seed(Element::Face))};
    auto classes = meshes.EncodeCornerClassification(r, chain, std::array{MeshStore::CornerClassRequest{id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Counts[1], source == Element::None}});
    chain.Submit();
    meshes.PlanCornerClassification(r, chain, classes);
    change.Triangles.Finish(chain);
    EncodeDeriveMeshNormals(r, chain, std::array{LocalNormalWork{id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]}});
    chain.Submit();
    MeshStore::SelectionUpdate update{.StoreId = id};
    for (const auto [d, domain, work] : {std::tuple{0u, MeshStore::ElementDomain::Vertex, 0u}, std::tuple{1u, MeshStore::ElementDomain::Edge, 3u}, std::tuple{2u, MeshStore::ElementDomain::Face, 2u}}) {
        update.Blocks[d] = meshes.ArenaBlocks(id, domain, chain.Scratch, neighborhood.Elements[work]);
    }
    meshes.UpdateSelection(r, chain, std::array{update});
    chain.Submit();
    return change;
}
