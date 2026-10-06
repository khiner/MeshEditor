#include "mesh/RecalculateNormals.h"
#include "Profile.h"
#include "gpu/RecalculateNormalsPushConstants.h"
#include "mesh/Mesh.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"
#include <unordered_set>

std::vector<std::vector<uint32_t>> RecalculateFaceFlips(state::Scene &r, std::span<const uint32_t> ids, bool inside) {
    if (ids.empty()) return {};
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    struct Job {
        RecalculateNormalsPushConstants Pc;
        std::vector<uvec2> Faces, Selected; // (handle, parity), (face rank, group)
        uint32_t Groups{}, Tiles{};
    };
    std::vector<Job> jobs;
    uint64_t source_faces = 0u, source_corners = 0u;
    for (const auto id : ids) {
        const Mesh mesh{meshes, id};
        Job job;
        std::vector<uvec4> groups;
        std::vector<uvec2> tiles;
        std::unordered_set<uint32_t> visited;
        MeshEdgeUsers edges{mesh};
        const auto selection = meshes.GetSelectedElements(id, Element::Face);
        selection.ForEach([&](uint32_t start) {
            if (!visited.insert(start).second) return;
            const uint32_t first = uint32_t(job.Faces.size()), group = uint32_t(groups.size());
            job.Faces.push_back({start, 0u});
            for (uint32_t at = first; at < job.Faces.size(); ++at) {
                const auto entry = job.Faces[at];
                if (selection.Contains(entry.x)) job.Selected.push_back({at, group});
                for (const auto h : mesh.fh_range(he::FH{entry.x})) {
                    ++source_corners;
                    const auto users = edges.Get(h);
                    if (users.Count != 2u) continue;
                    const he::HH other{users.First == *h ? users.Second : users.First};
                    const auto face = *mesh.GetFace(other);
                    if (!visited.insert(face).second) continue;
                    const uint32_t parity = entry.y ^ uint32_t(mesh.GetToVertex(h) == mesh.GetToVertex(other));
                    job.Faces.push_back({face, parity});
                }
            }
            const uint32_t count = uint32_t(job.Faces.size()) - first, tile = uint32_t(tiles.size());
            for (uint32_t i = 0u; i < count; i += 256u) tiles.push_back({group, first + i});
            groups.push_back({first, count, tile, uint32_t(tiles.size()) - tile});
        });
        source_faces += job.Faces.size();
        const auto allocate = [&](uint32_t words) { return SlotOffset{chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(words).Offset}; };
        job.Groups = uint32_t(groups.size());
        job.Tiles = uint32_t(tiles.size());
        job.Pc = {.Connectivity = meshes.GetConnectivityRef(id), .Faces = chain.Upload(as_bytes(job.Faces)), .Groups = chain.Upload(as_bytes(groups)), .Tiles = chain.Upload(as_bytes(tiles)), .PartialCenters = allocate(4u * job.Tiles), .Centers = allocate(3u * job.Groups), .Candidates = allocate(5u * job.Tiles), .Orientations = allocate(job.Groups), .VertexSlot = meshes.Slots().Vertices, .CornerSlot = meshes.Arenas().FaceCorners.Buffer.Slot, .NormalSlot = meshes.Arenas().BaseFaceNormals.Buffer.Slot, .FaceCount = mesh.FaceCount()};
        jobs.push_back(std::move(job));
    }
    if (!source_faces) return std::vector<std::vector<uint32_t>>(ids.size());
    const auto &pipelines = GetMeshPipelines(r);
    for (const auto pass : {MeshPass::RecalculateNormalFaceCenters, MeshPass::RecalculateNormalCenters, MeshPass::RecalculateNormalCandidates, MeshPass::RecalculateNormalOrientation})
        chain.Concurrent([&] {
            for (const auto &job : jobs) chain.Groups(pipelines[pass], job.Pc, pass == MeshPass::RecalculateNormalFaceCenters || pass == MeshPass::RecalculateNormalCandidates ? job.Tiles : job.Groups);
        });
    chain.Submit();
    std::vector<std::vector<uint32_t>> result;
    for (const auto &job : jobs) {
        auto &faces = result.emplace_back();
        const auto orientations = chain.Scratch.Get({job.Pc.Orientations.Offset, job.Groups});
        for (const auto selected : job.Selected) {
            const auto face = job.Faces[selected.x];
            if (face.y ^ orientations[selected.y] ^ uint32_t(inside)) faces.push_back(face.x);
        }
    }
    profile::RecordCounter("NormalOrientationFaces", source_faces);
    profile::RecordCounter("NormalOrientationCorners", source_corners);
    return result;
}
