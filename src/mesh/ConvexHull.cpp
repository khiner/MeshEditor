#include "mesh/ConvexHull.h"

#include "Profile.h"
#include "gpu/ConvexHull.h"
#include "mesh/MeshPipelines.h"
#include "mesh/ScratchPasses.h"

#include <algorithm>

std::vector<std::vector<std::vector<uint32_t>>> ConvexHulls(state::Registry &r, mtl::ComputeChain &chain, uint32_t positions, std::span<const uint32_t> first_point) {
    const profile::CpuScope scope{"ConvexHulls"};
    const auto sets = uint32_t(first_point.size() - 1u), points = first_point.back();
    std::vector<std::vector<std::vector<uint32_t>>> hulls(sets);
    if (!points) return hulls;
    const auto &pipelines = GetMeshPipelines(r);
    auto &scratch = chain.Scratch;
    constexpr uint32_t FaceWords = sizeof(HullFace) / sizeof(uint32_t), SetWords = sizeof(HullSet) / sizeof(uint32_t);
    HullJob job{.Scratch = scratch.Buffer.Slot, .Sets = sets, .Points = points, .SetRecords = TakeScratch(chain, uint64_t(sets) * SetWords).Offset, .Positions = positions, .PointFace = TakeScratch(chain, points).Offset, .Proposals = TakeScratch(chain, 1u).Offset};
    {
        std::vector<HullSet> set_records(sets);
        for (uint32_t s = 0u; s < sets; ++s) {
            set_records[s] = {.First = first_point[s], .Status = first_point[s + 1u] - first_point[s] < 4u ? HullSetFlat : HullSetOpen, .Seed = {InvalidOffset, InvalidOffset, InvalidOffset}};
        }
        scratch.Buffer.Update(std::as_bytes(std::span{set_records}), uint64_t(job.SetRecords) * sizeof(uint32_t));
        std::ranges::fill(scratch.GetMutable({job.PointFace, points}), HullPointUnassigned);
    }
    // Each pass takes the job as it stands when the pass records.
    const auto propose = [&] {
        scratch.GetMutable({job.Proposals, 1u})[0] = 0u;
        chain.Threads(pipelines[MeshPass::HullPropose], job, points);
        chain.Threads(pipelines[MeshPass::HullElect], job, points);
        chain.Threads(pipelines[MeshPass::HullClaim], job, job.Faces);
        chain.Threads(pipelines[MeshPass::HullAccept], job, job.Faces);
        chain.Threads(pipelines[MeshPass::HullHorizons], job, job.Faces + 1u);
        ScanScratch(chain, pipelines, std::array{Range{job.NewCount, job.Faces + 1u}, Range{job.Survivor, job.Faces + 1u}});
    };
    auto faces = TakeScratch(chain, 4ull * sets * FaceWords);
    auto new_count = TakeScratch(chain, uint64_t(4u * sets) + 1u), survivor = TakeScratch(chain, uint64_t(4u * sets) + 1u);
    job.Faces = 4u * sets;
    job.Current = job.Next = faces.Offset;
    job.NewCount = new_count.Offset;
    job.Survivor = survivor.Offset;
    for (job.Stage = 0u; job.Stage < 3u; ++job.Stage) {
        chain.Threads(pipelines[MeshPass::HullSeedMax], job, points);
        chain.Threads(pipelines[MeshPass::HullSeedElect], job, points);
    }
    chain.Threads(pipelines[MeshPass::HullSeedFaces], job, sets);
    chain.Threads(pipelines[MeshPass::HullAssign], job, points);
    propose();
    std::vector<Range> retired;
    uint32_t rounds = 1u;
    for (;; ++rounds) {
        chain.Submit();
        scratch.Release(std::move(retired));
        retired.clear();
        if (!scratch.Get({job.Proposals, 1u})[0]) break;
        job.NewFaces = scratch.Get({job.NewCount + job.Faces, 1u})[0];
        job.Survivors = scratch.Get({job.Survivor + job.Faces, 1u})[0];
        const auto next_faces = job.Survivors + job.NewFaces;
        auto next = TakeScratch(chain, uint64_t(next_faces) * FaceWords);
        job.Next = next.Offset;
        chain.Concurrent([&] {
            chain.Threads(pipelines[MeshPass::HullScatter], job, job.Faces);
            chain.Threads(pipelines[MeshPass::HullEmit], job, job.Faces);
        });
        chain.Threads(pipelines[MeshPass::HullAssign], job, points);
        retired = {faces, new_count, survivor};
        faces = next;
        new_count = TakeScratch(chain, uint64_t(next_faces) + 1u);
        survivor = TakeScratch(chain, uint64_t(next_faces) + 1u);
        job.Faces = next_faces;
        job.Current = faces.Offset;
        job.NewCount = new_count.Offset;
        job.Survivor = survivor.Offset;
        propose();
    }
    profile::RecordCounter("HullRounds", rounds);
    const auto set_records = ReadScratch<HullSet>(chain, job.SetRecords, sets);
    const auto hull_faces = ReadScratch<HullFace>(chain, job.Current, job.Faces);
    // A planar set's lateral faces hold its apex, and their other two vertices in order are the plane hull's edges, which chain into its polygon.
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edges(sets);
    for (uint32_t f = 0u; f < job.Faces; ++f) {
        const auto &face = hull_faces[f];
        const auto &set = set_records[face.Set];
        const std::array v{face.V.x, face.V.y, face.V.z};
        if (set.Status == HullSetOpen) hulls[face.Set].push_back({v[0] - set.First, v[1] - set.First, v[2] - set.First});
        else if (set.Status == HullSetPlanar) {
            for (uint32_t k = 0u; k < 3u; ++k) {
                if (v[k] == points + face.Set) edges[face.Set].emplace_back(v[(k + 1u) % 3u] - set.First, v[(k + 2u) % 3u] - set.First);
            }
        }
    }
    for (uint32_t s = 0u; s < sets; ++s) {
        if (edges[s].empty()) continue;
        std::ranges::sort(edges[s]);
        std::vector<uint32_t> loop{edges[s].front().first};
        while (loop.size() <= edges[s].size()) {
            const auto next = std::ranges::lower_bound(edges[s], loop.back(), {}, &std::pair<uint32_t, uint32_t>::first);
            if (next == edges[s].end() || next->first != loop.back() || next->second == loop.front()) break;
            loop.push_back(next->second);
        }
        hulls[s].push_back(std::move(loop));
    }
    return hulls;
}
