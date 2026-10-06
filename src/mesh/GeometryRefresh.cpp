#include "mesh/GeometryRefresh.h"
#include "Profile.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "state/Scene.h"

std::vector<FaceTriangles> RetessellateMeshes(state::Scene &r, mtl::ComputeChain &chain, std::span<const MeshRetessellation> inputs) {
    auto &meshes = r.Context.get<MeshStore>();
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
        const auto tangents = a.CornerTangents.Ref(!input.Preview && (meshes.Get(input.StoreId).CornerAttributes & MeshAttributeBit_Tangent));
        Job job{.Index = i, .Pc = input.Parameters};
        job.Pc.Tangents = tangents;
        job.Pc.VertexSlot = meshes.Slots().Vertices;
        job.Pc.CornerSlot = a.FaceCorners.Buffer.Slot;
        job.Pc.FaceRangesSlot = a.FaceRanges.Buffer.Slot;
        job.Pc.FaceTrianglesSlot = a.FaceTriangles.Buffer.Slot;
        job.Pc.TriangleSlot = a.Triangles.Buffer.Slot;
        std::vector<Range> triangle_ranges;
        ForEachWorkElement(*input.Work, input.Faces, [&](uint32_t face) {
            const auto loop = a.FaceRanges.Get({face, 1u})[0];
            const uint32_t n = loop.y - loop.x;
            if (n <= 3u && tangents.ValuesSlot == InvalidSlot) return;
            if (uint64_t(job.Corners) + n > UINT32_MAX / 4u) throw std::length_error("Polygon tessellation scratch exceeds its address space.");
            if (tangents.ValuesSlot != InvalidSlot) a.CornerTangents.CaptureHandles({loop.x, n});
            job.Faces.push_back({face, job.Corners});
            const uint32_t first = a.FaceTriangles.Get({face, 1u})[0];
            if (n > 3u) {
                job.Corners += n;
                triangle_ranges.push_back({first, n - 2u});
            }
            job.TriangleBlocks += (uint64_t(first % MeshElementBlockSize) + n - 2u + MeshElementBlockSize - 1u) / MeshElementBlockSize;
        });
        if (job.Faces.empty()) continue;
        a.Triangles.Buffer.CaptureWriteRanges(triangle_ranges, sizeof(uvec3));
        job.Pc.Count = uint32_t(job.Faces.size());
        const auto capacity = WorkCapacity(a.Triangles.Capacity(), job.TriangleBlocks);
        words += 4ull * job.Corners + 2ull * job.Faces.size() + ElementWorkWords(a.Triangles.Capacity(), job.TriangleBlocks) + SortElementWorkWords(capacity);
        jobs.push_back(std::move(job));
    }
    std::vector<FaceTriangles> results(inputs.size());
    if (jobs.empty()) return results;
    const profile::CpuScope scope{"RetessellateGeometry"};
    chain.Scratch.ReserveAdditional(words);
    std::vector<ElementWork> changed;
    uint64_t faces = 0u, corners = 0u;
    for (auto &job : jobs) {
        job.Pc.Faces = chain.Upload(as_bytes(job.Faces));
        job.Pc.Scratch = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(4u * job.Corners).Offset};
        job.Pc.ChangedTriangles = AllocateElementWork(chain.Scratch, a.Triangles.Capacity(), job.TriangleBlocks);
        changed.push_back(job.Pc.ChangedTriangles);
        faces += job.Pc.Count;
        corners += job.Corners;
    }
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::RetessellateFaces];
    chain.Concurrent([&] { for (const auto &job:jobs) chain.Threads(pipeline,job.Pc,job.Pc.Count); });
    EncodeSortElementWork(r, chain, changed);
    chain.Submit();
    uint64_t triangles = 0u;
    for (const auto &job : jobs) {
        auto &result = results[job.Index];
        result.Triangles = job.Pc.ChangedTriangles;
        result.Count = ElementWorkCount(chain.Scratch, result.Triangles);
        triangles += result.Count;
    }
    profile::RecordCounter("RetessellatedFaces", faces);
    profile::RecordCounter("RetessellatedCorners", corners);
    profile::RecordCounter("RetessellatedTriangles", triangles);
    return results;
}

std::vector<PositionOperationChange> ExecuteGeometryPositions(state::Scene &r, std::span<const PositionOperationTarget> targets, PositionEditOp operation, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto changes = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, targets, operation, factor, repeat, options);
    chain.Submit();
    std::vector<MeshClosure> neighborhoods;
    std::vector<MeshRetessellation> tessellations;
    std::vector<MeshStore::SelectionUpdate> bounds;
    for (const auto &change : changes) {
        const auto id = targets[change.TargetIndex].StoreId;
        const auto vertices = ListSeed(r, chain, id, Element::Vertex, change.Vertices);
        const auto incident = EncodeVertexClosure(r, chain, id, vertices);
        const auto faces = AroundVertices(r, id, Element::Face, incident.Elements[2], vertices);
        neighborhoods.push_back(EncodePrimitiveClosure(r, chain, id, faces, {}, vertices));
        auto &blocks = bounds.emplace_back(MeshStore::SelectionUpdate{.StoreId = id}).Blocks[0];
        ForEachWorkBlock(chain.Scratch, vertices.Work, [&](uint32_t block, auto) { blocks.push_back(block); });
        tessellations.push_back({.StoreId = id, .Work = &chain.Scratch, .Faces = neighborhoods.back().Elements[2], .Parameters = {.ChangedVertices = vertices.Work}});
    }
    chain.Submit();
    std::vector<LocalNormalWork> normals;
    for (uint32_t i = 0u; i < changes.size(); ++i) {
        const auto id = targets[changes[i].TargetIndex].StoreId;
        auto &neighborhood = neighborhoods[i];
        neighborhood.Finish(chain);
        ForEachWorkBlock(chain.Scratch, neighborhood.Elements[2], [&](uint32_t block, auto) { bounds[i].Blocks[2].push_back(block); });
        normals.push_back({id, neighborhood.Elements[0], neighborhood.Counts[0], neighborhood.Elements[2], neighborhood.Counts[2]});
    }
    RetessellateMeshes(r, chain, tessellations);
    EncodeDeriveMeshNormals(r, chain, chain.Scratch, normals);
    meshes.UpdateSelection(r, chain, bounds);
    chain.Submit();
    return changes;
}
