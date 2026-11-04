#include "mesh/BooleanWork.h"

#include "Profile.h"
#include "gpu/MeshBoolean.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/ScratchChunks.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

BooleanWork::BooleanWork(state::Registry &r, mtl::ComputeChain &chain, const MeshTopologyTask &task, const MeshClosure &first, const MeshClosure &second) {
    const profile::CpuScope scope{"BooleanWork"};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    auto &scratch = chain.Scratch;
    // A closed surface's corners all belong to its faces, which derive two fewer triangles than corners.
    const std::array triangles{first.Counts[1] - 2u * first.Counts[2], second.Counts[1] - 2u * second.Counts[2]};
    BooleanJob job{
        .Operation = BooleanOperation(uint32_t(task.Param0)),
        .VertexWork = {first.Elements[0], second.Elements[0]},
        .FaceWork = {first.Elements[2], second.Elements[2]},
        .Vertices = {first.Counts[0], second.Counts[0]},
        .Faces = {first.Counts[2], second.Counts[2]},
        .Triangles = triangles,
        .VertexSlot = a.Vertices.Buffer.Slot,
        .TriangleSlot = a.Triangles.Buffer.Slot,
        .FaceTriangleStartSlot = a.FaceTriangles.Buffer.Slot,
        .VertexNormalSlot = a.BaseVertexNormals.Buffer.Slot,
        .FaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
        .Connectivity = meshes.GetConnectivityRef(task.SourceId),
        .Rotation = task.CopyRotation,
        .Translation = task.CopyTranslation,
    };
    Vertices = job.Vertices[0] + job.Vertices[1];
    Triangles = triangles[0] + triangles[1];
    const uint32_t faces = job.Faces[0] + job.Faces[1], halfedges = 3u * Triangles;
    const auto take = [&](uint64_t words) { return TakeScratch(chain, words).Offset; };
    const auto record = take(sizeof(BooleanJob) / sizeof(uint32_t));
    Job = {scratch.Buffer.Slot, record};
    const auto upload = [&] { scratch.Buffer.Update(std::as_bytes(std::span{&job, 1u}), uint64_t(record) * sizeof(uint32_t)); };
    const JobPushConstants pc{.Job = Job};

    {
        const profile::CpuScope stage{"BooleanIntersect"};
        scratch.PlanAdditional(uint32_t(std::min<uint64_t>(2ull * Vertices + 67ull * Triangles + 1u, UINT32_MAX)));
        const auto allocate = [&](auto &&alloc) {
            job.Positions = alloc(3ull * Vertices);
            job.Displacements = alloc(3ull * Vertices);
            job.Bounds = alloc(12u);
            job.FaceTriangleFirst = alloc(uint64_t(faces) + 1u);
            job.TriangleVertices = alloc(3ull * Triangles);
            job.TriangleCorners = alloc(3ull * Triangles);
            job.TriangleFaces = alloc(Triangles);
            job.TriangleNormals = alloc(3ull * Triangles);
            job.PolygonTriangleCounts = alloc(Triangles);
            job.Pairs = alloc(halfedges);
            job.EdgeSources = alloc(halfedges);
            for (uint32_t side = 0u; side < 2u; ++side) job.Colliders[side] = AllocateCollider(alloc, triangles[side]);
            job.IntersectionFirst = alloc(uint64_t(halfedges) + 1u);
            job.Labels = alloc(Vertices);
            job.Windings = alloc(Vertices);
            job.LabelArguments = alloc(7u);
            job.Cursors = alloc(Triangles);
        };
        allocate([&](uint64_t words) { scratch.PlanAdditional(uint32_t(words)); return 0u; });
        scratch.CommitPlanned();
        allocate(take);
        const auto bounds = scratch.GetMutable({job.Bounds, 12u});
        for (uint32_t side = 0u; side < 2u; ++side) {
            std::ranges::fill(bounds.subspan(6u * side, 3u), ~0u);
            std::ranges::fill(bounds.subspan(6u * side + 3u, 3u), 0u);
        }
        upload();
        chain.Threads(pipelines[MeshPass::BooleanVertices], pc, Vertices);
        chain.Threads(pipelines[MeshPass::BooleanFaceCounts], pc, faces + 1u);
        ScanScratch(chain, pipelines, job.FaceTriangleFirst, faces + 1u);
        chain.Threads(pipelines[MeshPass::BooleanTriangles], pc, faces);
        chain.Threads(pipelines[MeshPass::BooleanPairs], pc, halfedges);
        chain.Threads(pipelines[MeshPass::BooleanMorton], pc, Triangles);
        const auto leaf_groups = TileCount(std::max(triangles[0], triangles[1]), TileElements);
        RecordColliderBuild(chain, pipelines, {scratch.Buffer.Slot, record + uint32_t(offsetof(BooleanJob, Colliders) / sizeof(uint32_t))}, leaf_groups, 2u);
        chain.Groups(pipelines[MeshPass::BooleanRefit], pc, leaf_groups, 256u, 2u);
        chain.Threads(pipelines[MeshPass::BooleanIntersectCount], pc, halfedges + 1u);
        ScanScratch(chain, pipelines, job.IntersectionFirst, halfedges + 1u);
        chain.Submit();
        const auto intersection_first = scratch.Get({job.IntersectionFirst, halfedges + 1u});
        Intersections = job.Intersections = intersection_first[halfedges];
        job.FirstIntersections = intersection_first[3u * triangles[0]];
        profile::RecordCounter("BooleanIntersections", Intersections);
    }
    {
        const profile::CpuScope stage{"BooleanComponents"};
        job.IntersectionRecords = take(6ull * Intersections);
        job.Inclusion = take(uint64_t(Vertices) + Intersections);
        upload();
        chain.Threads(pipelines[MeshPass::BooleanIntersectEmit], pc, halfedges);
        chain.Threads(pipelines[MeshPass::BooleanLabelInit], pc, Vertices);
        const uint64_t arguments_bytes = uint64_t(job.LabelArguments) * sizeof(uint32_t);
        for (uint32_t rounds = LabelRounds;; rounds *= 2u) {
            const auto arguments = scratch.GetMutable({job.LabelArguments, 7u});
            const std::array<uint32_t, 7> armed{TileCount(halfedges, TileElements), 1u, 1u, TileCount(Vertices, TileElements), 1u, 1u, 0u};
            std::ranges::copy(armed, arguments.begin());
            for (uint32_t round = 0u; round < rounds; ++round) {
                chain.Indirect(pipelines[MeshPass::BooleanLink], pc, scratch.Buffer, arguments_bytes);
                chain.Indirect(pipelines[MeshPass::BooleanJump], pc, scratch.Buffer, arguments_bytes + 3u * sizeof(uint32_t));
                chain.Groups(pipelines[MeshPass::BooleanConverge], pc, 1u, 32u);
            }
            chain.Submit();
            profile::RecordCounter("BooleanLabelRounds", rounds);
            if (!scratch.Get({job.LabelArguments, 1u})[0]) break;
        }
    }
    {
        const profile::CpuScope stage{"BooleanCounts"};
        const uint64_t count_entries = 4ull * Triangles + 2ull * Intersections + Vertices + Intersections;
        job.Counts = take(count_entries + 1u);
        upload();
        chain.Threads(pipelines[MeshPass::BooleanWinding], pc, Vertices);
        chain.Threads(pipelines[MeshPass::BooleanInclusion], pc, Vertices + Intersections);
        chain.Threads(pipelines[MeshPass::BooleanCounts], pc, uint32_t(count_entries + 1u));
        chain.Threads(pipelines[MeshPass::BooleanIntersectionSides], pc, Intersections);
        ScanScratch(chain, pipelines, job.Counts, uint32_t(count_entries + 1u));
        chain.Submit();
        const auto counts = scratch.Get({job.Counts, uint32_t(count_entries + 1u)});
        job.PolygonHalfedges = counts[Triangles];
        job.PartialEntries = counts[4u * Triangles] - job.PolygonHalfedges;
        job.GroupEntries = counts[4u * Triangles + 2u * Intersections] - job.PolygonHalfedges - job.PartialEntries;
        OutputVertices = counts[count_entries] - job.PolygonHalfedges - job.PartialEntries - job.GroupEntries;
    }
    {
        const profile::CpuScope stage{"BooleanPolygons"};
        const uint64_t polygon_halfedges = job.PolygonHalfedges;
        const auto allocate = [&](auto &&alloc) {
            job.OutputEntries = alloc(OutputVertices);
            job.Polygons = alloc(3u * polygon_halfedges);
            job.PartialEdges = alloc(4ull * job.PartialEntries);
            job.NewEdges = alloc(4ull * job.GroupEntries);
            job.LoopLinks = alloc(4u * polygon_halfedges);
            job.LoopNodes = alloc(7u * polygon_halfedges);
            job.LoopValues = alloc(6u * polygon_halfedges);
            job.PolygonTriangles = alloc(6u * polygon_halfedges);
            job.TriangleTotal = alloc(1u);
        };
        allocate([&](uint64_t words) { scratch.PlanAdditional(uint32_t(words)); return 0u; });
        scratch.CommitPlanned();
        allocate(take);
        scratch.GetMutable({job.TriangleTotal, 1u})[0] = 0u;
        upload();
        chain.Threads(pipelines[MeshPass::BooleanOutputs], pc, std::max(Vertices + Intersections, Triangles));
        // Each pass appends to the polygons through their atomic cursors.
        chain.Concurrent([&] {
            chain.Threads(pipelines[MeshPass::BooleanWholeEdges], pc, halfedges);
            chain.Threads(pipelines[MeshPass::BooleanPartialEdges], pc, halfedges);
            chain.Threads(pipelines[MeshPass::BooleanNewEdges], pc, 2u * Intersections);
        });
        chain.Threads(pipelines[MeshPass::BooleanFaces], pc, Triangles);
        chain.Submit();
        OutputTriangles = scratch.Get({job.TriangleTotal, 1u})[0];
        profile::RecordCounter("BooleanPolygonHalfedges", job.PolygonHalfedges);
        profile::RecordCounter("BooleanOutputTriangles", OutputTriangles);
    }
}
