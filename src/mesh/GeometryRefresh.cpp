#include "mesh/GeometryRefresh.h"
#include "Profile.h"
#include "gpu/PolygonTriangulation.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Registry.h"

std::vector<ElementWork> EncodeRetessellation(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshRetessellation> inputs) {
    const auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    struct Job {
        uint32_t Index;
        RetessellateFacesPushConstants Pc;
        std::vector<uvec2> Faces;
        uint32_t Corners{};
        uint64_t TriangleBlocks{};
    };
    std::vector<Job> jobs;
    uint64_t words = 0u;
    for (uint32_t i = 0u; i < inputs.size(); ++i) {
        const auto &input = inputs[i];
        const auto &record = meshes.Get(input.StoreId);
        const auto tangents = a.CornerTangents.Ref(record.CornerTangents);
        Job job{.Index = i, .Pc = input.Parameters};
        const auto face_table = a.FaceTriangles.Entries(record.FaceData);
        job.Pc.Tables = meshes.GetElementTables(input.StoreId);
        job.Pc.Tangents = tangents;
        job.Pc.VertexSlot = meshes.Slots().Vertices;
        job.Pc.CornerSlot = a.FaceCorners.Buffer.Slot;
        job.Pc.FaceRangesSlot = a.FaceRanges.Buffer.Slot;
        job.Pc.FaceTrianglesSlot = a.FaceTriangles.Buffer.Slot;
        job.Pc.TriangleSlot = a.Triangles.Buffer.Slot;
        std::vector<Range> triangle_ranges;
        ForEachWorkElement(*input.Work, input.Faces, [&](uint32_t face) {
            const auto address = Resolve<MeshElementBlockSize>(face_table, ElementHandle{face}).Value();
            const auto loop = a.FaceRanges.Get({address, 1u})[0];
            const uint32_t n = loop.y - loop.x;
            if (n <= 3u && tangents.Slot == InvalidSlot) return;
            if (uint64_t(job.Corners) + n > UINT32_MAX / (2u * PolygonTriangulationWords)) throw std::length_error("Polygon tessellation scratch exceeds its address space.");
            if (tangents.Slot != InvalidSlot) a.CornerTangents.Capture(record.CornerTangents, {loop.x, n});
            job.Faces.emplace_back(face, job.Corners);
            const uint32_t first = a.FaceTriangles.Get({address, 1u})[0];
            if (n > 3u) a.Triangles.ForEachAddressRun(record.TriangleData, ElementHandle{first}, n - 2u, [&](Range run) { triangle_ranges.push_back(run); });
            // Polygons beyond quads triangulate in scratch, and quads triangulate in thread memory.
            if (n > 4u) job.Corners += n;
            job.TriangleBlocks += (uint64_t(first % MeshElementBlockSize) + n - 2u + MeshElementBlockSize - 1u) / MeshElementBlockSize;
        });
        if (job.Faces.empty()) continue;
        a.Triangles.Buffer.CaptureWriteRanges(triangle_ranges, sizeof(uvec3));
        job.Pc.Count = uint32_t(job.Faces.size());
        const auto bound = a.Triangles.HandleBound(record.TriangleData);
        words += 2ull * PolygonTriangulationWords * job.Corners + 2ull * job.Faces.size() + ElementWorkWords(bound, job.TriangleBlocks) + SortElementWorkWords(WorkCapacity(bound, job.TriangleBlocks));
        jobs.push_back(std::move(job));
    }
    std::vector<ElementWork> results(inputs.size());
    if (jobs.empty()) return results;
    const profile::CpuScope scope{"RetessellateGeometry"};
    chain.Scratch.ReserveAdditional(words);
    std::vector<ElementWork> changed;
    uint64_t faces = 0u, corners = 0u;
    for (auto &job : jobs) {
        job.Pc.Faces = chain.Upload(BytesOf(job.Faces));
        job.Pc.Scratch = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * PolygonTriangulationWords * job.Corners).Offset};
        job.Pc.ChangedTriangles = AllocateElementWork(chain.Scratch, a.Triangles.HandleBound(meshes.Get(inputs[job.Index].StoreId).TriangleData), job.TriangleBlocks);
        changed.push_back(job.Pc.ChangedTriangles);
        results[job.Index] = job.Pc.ChangedTriangles;
        faces += job.Pc.Count;
        corners += job.Corners;
    }
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::RetessellateFaces];
    chain.Concurrent([&] { for (const auto &job:jobs) chain.Threads(pipeline,job.Pc,job.Pc.Count); });
    EncodeSortElementWork(r, chain, changed);
    profile::RecordCounter("RetessellatedFaces", faces);
    profile::RecordCounter("RetessellatedCorners", corners);
    return results;
}

std::optional<PositionOperationChange> ExecuteGeometryPositions(state::Registry &r, const PositionOperationTarget &target, PositionEditOp operation, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto change = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, target, operation, factor, repeat, options);
    chain.Submit();
    if (!change) return {};
    const auto id = target.StoreId;
    const auto vertices = ListSeed(r, chain, id, Element::Vertex, change->Vertices);
    const auto incident = EncodeVertexClosure(r, chain, id, vertices);
    const auto faces = AroundVertices(r, id, Element::Face, incident.Elements[2], vertices);
    auto neighborhood = EncodePrimitiveClosure(r, chain, id, faces, {}, vertices);
    MeshStore::SelectionUpdate bounds{.StoreId = id};
    bounds.Blocks[0] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Vertex, chain.Scratch, vertices.Work);
    const MeshRetessellation tessellation{.StoreId = id, .Work = &chain.Scratch, .Faces = neighborhood.Elements[2], .Parameters = {.ChangedVertices = vertices.Work}};
    chain.Submit();
    neighborhood.Finish(chain);
    bounds.Blocks[2] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Face, chain.Scratch, neighborhood.Elements[2]);
    EncodeRetessellation(r, chain, std::span{&tessellation, 1u});
    EncodeDeriveMeshNormals(r, chain, std::array{LocalNormalWork{id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]}});
    meshes.UpdateSelection(r, chain, std::array{bounds});
    chain.Submit();
    return change;
}
