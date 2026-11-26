#include "mesh/BooleanWork.h"

#include "Profile.h"
#include "gpu/MeshBoolean.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

std::vector<BooleanWork> RecordBooleanWork(state::Registry &r, mtl::ComputeChain &chain, std::span<const BooleanSources> sources) {
    const profile::CpuScope scope{"BooleanWork"};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    auto &scratch = chain.Scratch;
    const auto take = [&](uint64_t words) { return TakeScratch(chain, words).Offset; };
    std::vector<BooleanWork> works;
    std::vector<BooleanJob> jobs;
    works.reserve(sources.size());
    jobs.reserve(sources.size());
    for (const auto &[task, first, second] : sources) {
        // A closed surface's corners all belong to its faces, which derive two fewer triangles than corners.
        const std::array triangles{first.Counts[1] - 2u * first.Counts[2], second.Counts[1] - 2u * second.Counts[2]};
        const auto &job = jobs.emplace_back(BooleanJob{
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
        });
        works.push_back({.Vertices = job.Vertices[0] + job.Vertices[1], .Triangles = triangles[0] + triangles[1]});
    }
    const auto upload = [&](size_t i) { scratch.Buffer.Update(std::as_bytes(std::span{&jobs[i], 1u}), uint64_t(works[i].Job.Offset) * sizeof(uint32_t)); };
    const auto halfedges = [&](size_t i) { return 3u * works[i].Triangles; };
    const auto faces = [&](size_t i) { return jobs[i].Faces[0] + jobs[i].Faces[1]; };
    // Every pass runs over the jobs whose records lie in the scratch, each threadgroup finding its job through a table of first tiles.
    std::vector<SlotOffset> records;
    const auto tiles = [&](auto &&threads_of) { return std::views::iota(size_t{0}, works.size()) | std::views::transform([&](size_t i) { return TileCount(threads_of(i), JobTileThreads); }) | std::ranges::to<std::vector>(); };
    const auto dispatch = [&](MeshPass pass, const mtl::JobTable &table) { chain.Jobs(pipelines[pass], table, JobTileThreads); };
    // Records `fn(table, jobs)` with the table of the jobs that `threads_of(i)` threads of each job run over.
    const auto with_table = [&](MeshPass pass, auto &&threads_of, auto &&fn) { mtl::GatherJobs(chain, &pipelines[pass], std::span{records}, tiles(threads_of), fn); };
    const auto record = [&](MeshPass pass, auto &&threads_of) { mtl::GatherPass(chain, pipelines[pass], std::span{records}, tiles(threads_of), JobTileThreads); };
    const auto scan = [&](auto &&run_of) { ScanScratch(chain, pipelines, std::views::iota(size_t{0}, works.size()) | std::views::transform(run_of) | std::ranges::to<std::vector>()); };
    {
        const profile::CpuScope stage{"BooleanIntersect"};
        const auto allocate = [&](size_t i, auto &&alloc) {
            auto &job = jobs[i];
            const auto vertices = works[i].Vertices, triangles = works[i].Triangles;
            job.Positions = alloc(3ull * vertices);
            job.Displacements = alloc(3ull * vertices);
            job.Bounds = alloc(12u);
            job.FaceTriangleFirst = alloc(uint64_t(faces(i)) + 1u);
            job.TriangleVertices = alloc(3ull * triangles);
            job.TriangleCorners = alloc(3ull * triangles);
            job.TriangleFaces = alloc(triangles);
            job.TriangleNormals = alloc(3ull * triangles);
            job.PolygonTriangleCounts = alloc(triangles);
            job.Pairs = alloc(halfedges(i));
            job.EdgeSources = alloc(halfedges(i));
            for (uint32_t side = 0u; side < 2u; ++side) job.Colliders[side] = AllocateCollider(alloc, job.Triangles[side]);
            job.IntersectionFirst = alloc(uint64_t(halfedges(i)) + 1u);
            job.Labels = alloc(vertices);
            job.Windings = alloc(vertices);
            job.LabelState = alloc(2u);
            job.Cursors = alloc(triangles);
        };
        uint64_t words = 0u;
        for (size_t i = 0u; i < works.size(); ++i) {
            words += std::min<uint64_t>(2ull * works[i].Vertices + 67ull * works[i].Triangles + 1u, UINT32_MAX) + sizeof(BooleanJob) / sizeof(uint32_t) + BooleanTableWords;
            allocate(i, [&](uint64_t n) { words += n; return 0u; });
        }
        scratch.ReserveAdditional(words);
        for (size_t i = 0u; i < works.size(); ++i) {
            works[i].Job = {scratch.Buffer.Slot, take(sizeof(BooleanJob) / sizeof(uint32_t))};
            allocate(i, take);
            const auto bounds = scratch.GetMutable({jobs[i].Bounds, 12u});
            for (uint32_t side = 0u; side < 2u; ++side) {
                std::ranges::fill(bounds.subspan(6u * side, 3u), ~0u);
                std::ranges::fill(bounds.subspan(6u * side + 3u, 3u), 0u);
            }
            upload(i);
        }
        records = works | std::views::transform(&BooleanWork::Job) | std::ranges::to<std::vector>();
        record(MeshPass::BooleanVertices, [&](size_t i) { return works[i].Vertices; });
        record(MeshPass::BooleanFaceCounts, [&](size_t i) { return faces(i) + 1u; });
        scan([&](size_t i) { return Range{jobs[i].FaceTriangleFirst, faces(i) + 1u}; });
        record(MeshPass::BooleanTriangles, faces);
        record(MeshPass::BooleanPairs, halfedges);
        record(MeshPass::BooleanMorton, [&](size_t i) { return works[i].Triangles; });
        std::vector<ColliderBuild> colliders;
        for (size_t i = 0u; i < works.size(); ++i) {
            for (uint32_t side = 0u; side < 2u; ++side) colliders.push_back({works[i].Job.Offset + uint32_t((offsetof(BooleanJob, Colliders) + side * sizeof(ColliderRuns)) / sizeof(uint32_t)), jobs[i].Triangles[side]});
        }
        RecordColliderBuild(chain, pipelines, colliders);
        record(MeshPass::BooleanRefit, [&](size_t i) { return works[i].Triangles; });
        record(MeshPass::BooleanIntersectCount, [&](size_t i) { return halfedges(i) + 1u; });
        scan([&](size_t i) { return Range{jobs[i].IntersectionFirst, halfedges(i) + 1u}; });
        chain.Submit();
        for (size_t i = 0u; i < works.size(); ++i) {
            const auto intersection_first = scratch.Get({jobs[i].IntersectionFirst, halfedges(i) + 1u});
            works[i].Intersections = jobs[i].Intersections = intersection_first[halfedges(i)];
            jobs[i].FirstIntersections = intersection_first[3u * jobs[i].Triangles[0]];
            profile::RecordCounter("BooleanIntersections", works[i].Intersections);
        }
    }
    {
        const profile::CpuScope stage{"BooleanComponents"};
        for (size_t i = 0u; i < works.size(); ++i) {
            jobs[i].IntersectionRecords = take(6ull * works[i].Intersections);
            jobs[i].Inclusion = take(uint64_t(works[i].Vertices) + works[i].Intersections);
            std::ranges::fill(scratch.GetMutable({jobs[i].LabelState, 2u}), 0u);
            upload(i);
        }
        // Each submit runs its rounds over every job, with twice the rounds of the submit before it, until every job's labels converge.
        with_table(MeshPass::BooleanLink, halfedges, [&](const mtl::JobTable &edges, auto) {
            with_table(MeshPass::BooleanJump, [&](size_t i) { return works[i].Vertices; }, [&](const mtl::JobTable &vertices, auto) {
                bool converged = false;
                for (uint32_t rounds = LabelRounds; !converged; rounds *= 2u) {
                    if (rounds == LabelRounds) {
                        dispatch(MeshPass::BooleanIntersectEmit, edges);
                        dispatch(MeshPass::BooleanLabelInit, vertices);
                    }
                    for (uint32_t round = 0u; round < rounds; ++round) {
                        dispatch(MeshPass::BooleanLink, edges);
                        dispatch(MeshPass::BooleanJump, vertices);
                        chain.Threads(pipelines[MeshPass::BooleanConverge], edges.Constants, uint32_t(works.size()));
                    }
                    chain.Submit();
                    profile::RecordCounter("BooleanLabelRounds", rounds);
                    converged = std::ranges::all_of(jobs, [&](const BooleanJob &job) { return scratch.Get({job.LabelState, 1u})[0] != 0u; });
                }
            });
        });
    }
    {
        const profile::CpuScope stage{"BooleanCounts"};
        const auto count_entries = [&](size_t i) { return 4ull * works[i].Triangles + 2ull * works[i].Intersections + works[i].Vertices + works[i].Intersections; };
        for (size_t i = 0u; i < works.size(); ++i) {
            jobs[i].Counts = take(count_entries(i) + 1u);
            upload(i);
        }
        record(MeshPass::BooleanWinding, [&](size_t i) { return works[i].Vertices; });
        record(MeshPass::BooleanInclusion, [&](size_t i) { return works[i].Vertices + works[i].Intersections; });
        record(MeshPass::BooleanCounts, [&](size_t i) { return count_entries(i) + 1u; });
        record(MeshPass::BooleanIntersectionSides, [&](size_t i) { return works[i].Intersections; });
        scan([&](size_t i) { return Range{jobs[i].Counts, uint32_t(count_entries(i) + 1u)}; });
        chain.Submit();
        for (size_t i = 0u; i < works.size(); ++i) {
            auto &job = jobs[i];
            const auto triangles = works[i].Triangles, intersections = works[i].Intersections;
            const auto counts = scratch.Get({job.Counts, uint32_t(count_entries(i) + 1u)});
            job.PolygonHalfedges = counts[triangles];
            job.PartialEntries = counts[4u * triangles] - job.PolygonHalfedges;
            job.GroupEntries = counts[4u * triangles + 2u * intersections] - job.PolygonHalfedges - job.PartialEntries;
            works[i].OutputVertices = counts[count_entries(i)] - job.PolygonHalfedges - job.PartialEntries - job.GroupEntries;
        }
    }
    {
        const profile::CpuScope stage{"BooleanPolygons"};
        const auto allocate = [&](size_t i, auto &&alloc) {
            auto &job = jobs[i];
            const uint64_t polygon_halfedges = job.PolygonHalfedges;
            job.OutputEntries = alloc(works[i].OutputVertices);
            job.Polygons = alloc(3u * polygon_halfedges);
            job.PartialEdges = alloc(4ull * job.PartialEntries);
            job.NewEdges = alloc(4ull * job.GroupEntries);
            job.LoopLinks = alloc(4u * polygon_halfedges);
            job.LoopNodes = alloc(7u * polygon_halfedges);
            job.LoopValues = alloc(6u * polygon_halfedges);
            job.PolygonTriangles = alloc(6u * polygon_halfedges);
            job.TriangleTotal = alloc(1u);
        };
        uint64_t words = 0u;
        for (size_t i = 0u; i < works.size(); ++i) allocate(i, [&](uint64_t n) { words += n; return 0u; });
        scratch.ReserveAdditional(words);
        for (size_t i = 0u; i < works.size(); ++i) {
            allocate(i, take);
            scratch.GetMutable({jobs[i].TriangleTotal, 1u})[0] = 0u;
            upload(i);
        }
        record(MeshPass::BooleanOutputs, [&](size_t i) { return std::max(works[i].Vertices + works[i].Intersections, works[i].Triangles); });
        // Each pass appends to the polygons through their atomic cursors.
        chain.Concurrent([&] {
            record(MeshPass::BooleanWholeEdges, halfedges);
            record(MeshPass::BooleanPartialEdges, halfedges);
            record(MeshPass::BooleanNewEdges, [&](size_t i) { return 2u * works[i].Intersections; });
        });
        record(MeshPass::BooleanFaces, [&](size_t i) { return works[i].Triangles; });
        chain.Submit();
        for (size_t i = 0u; i < works.size(); ++i) {
            works[i].OutputTriangles = scratch.Get({jobs[i].TriangleTotal, 1u})[0];
            profile::RecordCounter("BooleanPolygonHalfedges", jobs[i].PolygonHalfedges);
            profile::RecordCounter("BooleanOutputTriangles", works[i].OutputTriangles);
        }
    }
    return works;
}
