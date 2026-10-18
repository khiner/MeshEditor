#include "mesh/SharpnessOperations.h"
#include "gpu/EditSharpnessPushConstants.h"
#include "mesh/CornerClassBatch.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "state/Registry.h"
#include <unordered_set>

std::vector<SharpnessOperationChange> ExecuteGeometrySharpness(state::Registry &r, mtl::ComputeChain &chain, std::span<const SharpnessOperationTarget> targets, EditSharpnessOperation operation, bool value, float angle) {
    if (uint32_t(operation) > uint32_t(EditSharpnessOperation::SetVertexEdges) || !std::isfinite(angle)) return {};
    auto &meshes = r.Context.get<MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto source = operation == EditSharpnessOperation::SetSelectedFaces ? Element::Face :
        operation == EditSharpnessOperation::SetSelectedEdges                 ? Element::Edge :
        operation == EditSharpnessOperation::SetVertexEdges                   ? Element::Vertex :
                                                                                Element::None;
    std::unordered_set<uint32_t> ids;
    std::vector<SelectedHandles> selections;
    for (const auto &target : targets) {
        if (!ids.insert(target.StoreId).second) throw std::invalid_argument("Sharpness operations require distinct meshes.");
        ValidateGeometrySelection(meshes, target.StoreId, *selections.emplace_back(meshes, target.StoreId, target.Selection));
    }
    struct Edit {
        uint32_t TargetIndex, Id;
        EditSharpnessPushConstants Pc;
        ClosureSeed Vertices;
        MeshClosure Neighborhood;
        FaceTriangles Triangles;
    };
    std::vector<Edit> edits;
    std::vector<ElementWorkSeedJob> seeds;
    const auto &arenas = meshes.Arenas();
    for (uint32_t i = 0u; i < targets.size(); ++i) {
        const auto id = targets[i].StoreId;
        const auto &selection = *selections[i];
        const Mesh mesh{meshes, id};
        if (!mesh.FaceCount() || (source != Element::None && selection.Get(source).empty())) continue;
        meshes.CaptureSharpnessWrite(id, operation, selection);
        const auto &record = meshes.Get(id);
        auto &pc = edits.emplace_back(Edit{.TargetIndex = i, .Id = id}).Pc;
        pc = {
            .VertexSelection = source == Element::Vertex ? SeedElementWorkHandles(chain.Scratch, arenas.Vertices.HandleBound(record.Vertices), selection.Vertices) : ElementWork{},
            .CornersSlot = arenas.FaceCorners.Buffer.Slot,
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
        if (source != Element::None) {
            const auto &selected = selection.Get(source);
            pc.Selected = chain.Upload(std::as_bytes(std::span{selected}));
            pc.SelectedCount = uint32_t(selected.size());
        } else {
            pc.FaceWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.FaceTriangles, record.FaceData)).Work;
            if (operation != EditSharpnessOperation::SetAllFaces) pc.EdgeWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch, arenas.EdgeHalfedges, record.EdgeData)).Work;
        }
    }
    if (edits.empty()) return {};
    std::vector<ElementWork> seeded;
    seeded.reserve(seeds.size());
    for (const auto &seed : seeds) seeded.push_back(seed.Work);
    EncodeElementMembershipWork(r, chain, seeds);
    EncodeSortElementWork(r, chain, seeded);
    chain.Submit();
    for (const auto work : seeded) CheckElementWork(chain.Scratch, work);
    // The closures record after the sharpness writes, and the next submit commits both.
    chain.Concurrent([&] {
        for (const auto &edit : edits) {
            const auto &pc = edit.Pc;
            const uint32_t count = source != Element::None       ? pc.SelectedCount :
                operation == EditSharpnessOperation::SetAllFaces ? pc.FaceCount :
                                                                   std::max(pc.EdgeCount, pc.FaceCount);
            chain.Groups(pipelines[MeshPass::EditSharpness], pc, (count + 255u) / 256u);
        }
    });
    for (auto &edit : edits) {
        const auto &selection = *selections[edit.TargetIndex];
        const auto id = edit.Id;
        if (source == Element::Face) edit.Vertices = EncodePrimitiveClosure(r, chain, id, ListSeed(r, chain, id, Element::Face, selection.Faces)).Seed(Element::Vertex);
        else if (source == Element::Edge) edit.Vertices = EncodeEdgeVertices(r, chain, id, ListSeed(r, chain, id, Element::Edge, selection.Edges));
        else edit.Vertices = source == Element::None ? EncodeAllSeed(r, chain, id, Element::Vertex) : ListSeed(r, chain, id, Element::Vertex, selection.Vertices);
    }
    std::erase_if(edits, [](const Edit &edit) { return !edit.Vertices.Count; });
    if (operation == EditSharpnessOperation::SetVertexEdges) {
        // Each edge at a selected vertex changes the fans at both of its endpoints.
        for (auto &edit : edits) {
            edit.Neighborhood = EncodeVertexClosure(r, chain, edit.Id, edit.Vertices);
            edit.Neighborhood.EncodeIncidence(r, chain, edit.Id, Element::Edge);
        }
        chain.Submit();
        for (auto &edit : edits) {
            edit.Neighborhood.Finish(chain);
            edit.Vertices = EncodeEdgeVertices(r, chain, edit.Id, edit.Neighborhood.Seed(Element::Edge));
        }
    }
    for (auto &edit : edits) {
        edit.Neighborhood = EncodeVertexClosure(r, chain, edit.Id, edit.Vertices);
        edit.Neighborhood.EncodeIncidence(r, chain, edit.Id, Element::Face);
    }
    chain.Submit();
    for (auto &edit : edits) edit.Neighborhood.Finish(chain);
    std::erase_if(edits, [](const Edit &edit) { return !edit.Neighborhood.Counts[0]; });
    std::vector<MeshStore::CornerClassRequest> requests;
    for (auto &edit : edits) {
        edit.Triangles = EncodeFaceTriangles(r, chain, edit.Id, edit.Neighborhood.Seed(Element::Face));
        // The neighborhood's corners include every corner at its vertices.
        const auto &neighborhood = edit.Neighborhood;
        requests.push_back({edit.Id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Counts[1], source == Element::None});
    }
    auto classes = meshes.EncodeCornerClassification(r, chain, requests);
    chain.Submit();
    meshes.PlanCornerClassification(r, chain, classes);
    std::vector<LocalNormalWork> normals;
    for (auto &edit : edits) {
        edit.Triangles.Finish(chain);
        const auto &neighborhood = edit.Neighborhood;
        normals.push_back({edit.Id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]});
    }
    EncodeDeriveMeshNormals(r, chain, chain.Scratch, normals);
    chain.Submit();
    std::vector<SharpnessOperationChange> changes;
    std::vector<MeshStore::SelectionUpdate> updates;
    for (const auto &edit : edits) {
        changes.push_back({edit.TargetIndex, edit.Triangles});
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = edit.Id});
        for (const auto [d, domain, work] : {std::tuple{0u, MeshStore::ElementDomain::Vertex, 0u}, std::tuple{1u, MeshStore::ElementDomain::Edge, 3u}, std::tuple{2u, MeshStore::ElementDomain::Face, 2u}}) {
            update.Blocks[d] = meshes.ArenaBlocks(edit.Id, domain, chain.Scratch, edit.Neighborhood.Elements[work]);
        }
    }
    meshes.UpdateSelection(r, chain, updates);
    chain.Submit();
    return changes;
}
