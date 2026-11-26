#include "mesh/GeometryRefresh.h"
#include "Profile.h"
#include "gpu/PolygonTriangulation.h"
#include "gpu/RetessellateFacesJob.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Registry.h"

namespace {
// A job over a record's faces, reading its loops and positions and writing its triangles.
RetessellateFacesJob RecordJob(const MeshStore &meshes, uint32_t id) {
    const auto &a = meshes.Arenas();
    return {
        .Tables = meshes.GetElementTables(id),
        .VertexSlot = meshes.Slots().Vertices,
        .CornerSlot = a.FaceCorners.Buffer.Slot,
        .FaceRangesSlot = a.FaceRanges.Buffer.Slot,
        .FaceTrianglesSlot = a.FaceTriangles.Buffer.Slot,
        .TriangleSlot = a.Triangles.Buffer.Slot,
    };
}

void RecordJobs(state::Registry &r, mtl::ComputeChain &chain, std::span<const RetessellateFacesJob> jobs) {
    mtl::GatherPass(chain, GetMeshPipelines(r)[MeshPass::RetessellateFaces], jobs, &RetessellateFacesJob::Count);
}
} // namespace

std::vector<ElementWork> EncodeRetessellation(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshRetessellation> inputs) {
    const auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    struct Job {
        uint32_t Index;
        RetessellateFacesJob Gpu;
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
        Job job{.Index = i, .Gpu = RecordJob(meshes, input.StoreId)};
        const auto face_table = a.FaceTriangles.Entries(record.FaceData);
        job.Gpu.ChangedVertices = input.ChangedVertices;
        job.Gpu.Tangents = tangents;
        std::vector<Range> triangle_ranges;
        const auto add = [&](uint32_t face) {
            const auto address = Resolve<MeshElementBlockSize>(face_table, ElementHandle{face}).Value();
            const auto loop = a.FaceRanges.Get({address, 1u})[0];
            const uint32_t n = loop.y - loop.x;
            if (n <= 3u && tangents.Slot == InvalidSlot) return;
            if (uint64_t(job.Corners) + n > UINT32_MAX / (2u * PolygonTriangulationWords)) throw std::length_error("Polygon tessellation scratch exceeds its address space.");
            if (tangents.Slot != InvalidSlot) a.CornerTangents.Capture(record.CornerTangents, {loop.x, n});
            job.Faces.emplace_back(face, job.Corners);
            const uint32_t first = a.FaceTriangles.Get({address, 1u})[0];
            if (n > 3u) a.Triangles.ForEachAddressRun(record.TriangleData, ElementHandle{first}, n - 2u, [&](Range run) { triangle_ranges.push_back(run); });
            // Polygons beyond PolygonThreadCorners corners triangulate in scratch.
            if (n > PolygonThreadCorners) job.Corners += n;
            job.TriangleBlocks += (uint64_t(first % MeshElementBlockSize) + n - 2u + MeshElementBlockSize - 1u) / MeshElementBlockSize;
        };
        if (input.Work) ForEachWorkElement(*input.Work, input.Faces, add);
        else a.FaceTriangles.ForEach(record.FaceData, [&](ElementHandle face, ElementAddress) { add(face.Value); });
        if (job.Faces.empty()) continue;
        a.Triangles.Buffer.CaptureWriteRanges(triangle_ranges, sizeof(uvec3));
        job.Gpu.Count = uint32_t(job.Faces.size());
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
        job.Gpu.Faces = chain.Upload(BytesOf(job.Faces));
        job.Gpu.Scratch = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * PolygonTriangulationWords * job.Corners).Offset};
        job.Gpu.ChangedTriangles = AllocateElementWork(chain.Scratch, a.Triangles.HandleBound(meshes.Get(inputs[job.Index].StoreId).TriangleData), job.TriangleBlocks);
        changed.push_back(job.Gpu.ChangedTriangles);
        results[job.Index] = job.Gpu.ChangedTriangles;
        faces += job.Gpu.Count;
        corners += job.Corners;
    }
    RecordJobs(r, chain, jobs | std::views::transform(&Job::Gpu) | std::ranges::to<std::vector>());
    EncodeSortElementWork(r, chain, changed);
    profile::RecordCounter("RetessellatedFaces", faces);
    profile::RecordCounter("RetessellatedCorners", corners);
    return results;
}

NewTriangles PlanNewTriangles(const MeshStore &meshes, uint32_t id) {
    const auto &record = meshes.Get(id);
    const auto &a = meshes.Arenas();
    NewTriangles plan{.StoreId = id};
    // A record without face starts holds only triangles.
    if (!record.ConnectivityFaceStarts) return plan;
    const auto loops = a.FaceRanges.Get(a.FaceTriangles.Dense(record.FaceData));
    for (uint32_t face = 0u; face < loops.size(); ++face) {
        const uint32_t n = loops[face].y - loops[face].x;
        if (n <= PolygonThreadCorners) continue;
        plan.Polygons.emplace_back(face, plan.Corners);
        plan.Corners += n;
    }
    plan.Words = 2ull * PolygonTriangulationWords * plan.Corners + 2ull * plan.Polygons.size();
    return plan;
}

void EncodeNewTriangles(state::Registry &r, mtl::ComputeChain &chain, std::span<const NewTriangles> plans) {
    const auto &meshes = r.Resources.get<MeshStore>();
    std::vector<RetessellateFacesJob> faces, polygons;
    uint64_t words = 0u;
    for (const auto &plan : plans) {
        auto job = RecordJob(meshes, plan.StoreId);
        job.Count = meshes.Arenas().FaceTriangles.Count(meshes.Get(plan.StoreId).FaceData);
        if (job.Count) faces.push_back(job);
        words += plan.Words;
    }
    chain.Scratch.ReserveAdditional(words);
    for (const auto &plan : plans) {
        if (plan.Polygons.empty()) continue;
        auto job = RecordJob(meshes, plan.StoreId);
        job.Faces = chain.Upload(BytesOf(plan.Polygons));
        job.Scratch = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * PolygonTriangulationWords * plan.Corners).Offset};
        job.Count = uint32_t(plan.Polygons.size());
        polygons.push_back(job);
    }
    // The polygon jobs read the first-triangle tables the face jobs write.
    if (!faces.empty()) RecordJobs(r, chain, faces);
    if (!polygons.empty()) RecordJobs(r, chain, polygons);
}

std::vector<ElementWork> EncodeRecordRefresh(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    using Domain = MeshStore::ElementDomain;
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<MeshRetessellation> tessellations;
    std::vector<MeshStore::SelectionUpdate> bounds;
    for (const auto id : ids) {
        // Work without storage covers every vertex handle.
        tessellations.push_back({.StoreId = id, .ChangedVertices = {.Count = meshes.Arenas().Vertices.HandleBound(meshes.Get(id).Vertices)}});
        auto &update = bounds.emplace_back(MeshStore::SelectionUpdate{.StoreId = id});
        update.Blocks[0].assign_range(meshes.GetBlockList(id, Domain::Vertex).Blocks);
        update.Blocks[2].assign_range(meshes.GetBlockList(id, Domain::Face).Blocks);
    }
    auto retessellated = EncodeRetessellation(r, chain, tessellations);
    EncodeDeriveRecordNormals(r, chain, ids);
    meshes.UpdateSelection(r, chain, bounds);
    return retessellated;
}

std::vector<ElementWork> EncodePositionRefresh(state::Registry &r, mtl::ComputeChain &chain, std::span<const PositionRefresh> refreshes) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<ElementWork> results(refreshes.size());
    std::vector<uint32_t> every;
    std::vector<size_t> every_at;
    // A record moved over listed vertices refreshes around them, with the neighborhood its closures record in its lane.
    struct Local {
        size_t At;
        uint32_t StoreId;
        ClosureSeed Vertices{};
        MeshClosure Neighborhood{};
        MeshStore::SelectionUpdate Bounds{};
    };
    std::vector<Local> locals;
    for (size_t i = 0u; i < refreshes.size(); ++i) {
        const auto [id, change] = refreshes[i];
        if (change->Every) {
            every.push_back(id);
            every_at.push_back(i);
            continue;
        }
        auto &local = locals.emplace_back(Local{.At = i, .StoreId = id, .Bounds = {.StoreId = id}});
        chain.Lane(id, [&] {
            local.Vertices = ListSeed(r, chain, id, Element::Vertex, change->Vertices);
            const auto incident = EncodeVertexClosure(r, chain, id, local.Vertices);
            const auto faces = AroundVertices(r, id, Element::Face, incident.Elements[2], local.Vertices);
            local.Neighborhood = EncodePrimitiveClosure(r, chain, id, faces, {}, local.Vertices);
        });
        local.Bounds.Blocks[0] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Vertex, chain.Scratch, local.Vertices.Work);
    }
    if (!locals.empty()) {
        chain.Submit();
        std::vector<MeshRetessellation> tessellations;
        std::vector<LocalNormalWork> normals;
        std::vector<MeshStore::SelectionUpdate> bounds;
        for (auto &local : locals) {
            auto &neighborhood = local.Neighborhood;
            neighborhood.Finish(chain);
            local.Bounds.Blocks[2] = meshes.ArenaBlocks(local.StoreId, MeshStore::ElementDomain::Face, chain.Scratch, neighborhood.Elements[2]);
            tessellations.push_back({.StoreId = local.StoreId, .Work = &chain.Scratch, .Faces = neighborhood.Elements[2], .ChangedVertices = local.Vertices.Work});
            normals.push_back({local.StoreId, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]});
            bounds.push_back(std::move(local.Bounds));
        }
        const auto retessellated = EncodeRetessellation(r, chain, tessellations);
        for (size_t k = 0u; k < locals.size(); ++k) results[locals[k].At] = retessellated[k];
        EncodeDeriveMeshNormals(r, chain, chain.Scratch, normals);
        meshes.UpdateSelection(r, chain, bounds);
    }
    if (!every.empty()) {
        const auto refreshed = EncodeRecordRefresh(r, chain, every);
        for (size_t k = 0u; k < every.size(); ++k) results[every_at[k]] = refreshed[k];
    }
    return results;
}

std::optional<PositionOperationChange> ExecuteGeometryPositions(state::Registry &r, const PositionOperationTarget &target, PositionEditOp operation, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto change = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, target, operation, factor, repeat, options);
    if (change) EncodePositionRefresh(r, chain, std::array{PositionRefresh{target.StoreId, &*change}});
    chain.Submit();
    return change;
}
