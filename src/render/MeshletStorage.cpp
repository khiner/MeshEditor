#include "render/MeshletStorage.h"
#include "render/GpuBufferOps.h"
#include "state/Scene.h"

void RetireMeshletStorage(state::Scene &r, MeshStore::Record &owner, std::span<const uint32_t> clusters) {
    if (clusters.empty()) return;
    if (owner.MeshletRoot == InvalidOffset) throw std::invalid_argument("Meshlet retirement requires owned canonical membership.");
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    uint32_t finest = 0u;
    for (const auto id : clusters) {
        if (id >= render.Meshlets.Buffer.Count<MeshletRecord>() || !render.ActiveMeshlets.Contains(owner.MeshletRoot, id)) {
            throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        }
        const auto &record = render.Meshlets.Get({id, 1u})[0];
        const bool refined = record.RefinedGroup != InvalidOffset;
        if (record.Topology > 2u || !record.VertexCount || record.VertexCount > 64u || !record.TriangleCount ||
            record.TriangleCount > (record.Topology ? 16u : 48u) ||
            (!refined && uint64_t(record.TriangleOffset) + record.TriangleCount > render.MeshletTriangleIds.Buffer.Count<uint32_t>()) ||
            uint64_t(record.VertexOffset) + record.VertexCount > render.MeshletVertexCorners.Buffer.Count<uint32_t>() ||
            (!record.Topology && uint64_t(record.LocalTriangleOffset) + record.TriangleCount * 3u > render.MeshletLocalTriangles.Buffer.Count<uint8_t>()))
            throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        finest += !refined;
    }
    if (finest > owner.Level0Count) throw std::logic_error("Meshlet retirement exceeds finest membership.");
    std::vector<uint32_t> sorted(clusters.begin(), clusters.end());
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end()) throw std::invalid_argument("Meshlet retirement repeats cluster identities.");
    MeshletIndexEdit edit{.Root = owner.MeshletRoot, .Removed = sorted};
    render.ActiveMeshlets.Update(std::span{&edit, 1u});
    owner.MeshletRoot = edit.Root;
    render.ReleaseMeshletStorage(sorted);
    owner.Level0Count -= finest;
    ++owner.MeshletRevision;
    UpdatePosedMeshletBlocks(r, owner, sorted);
}
