#include "render/MeshletStorage.h"
#include "render/GpuBuffers.h"
#include "state/Scene.h"

void RetireMeshletStorage(state::Scene &r, MeshBuffers &owner, std::span<const uint32_t> clusters) {
    if (clusters.empty()) return;
    if (owner.MeshletRoot == InvalidOffset) throw std::invalid_argument("Meshlet retirement requires owned canonical membership.");
    auto &buffers = r.Context.get<GpuBuffers>();
    uint32_t finest = 0u;
    for (const auto id : clusters) {
        if (id >= buffers.Meshlets.Buffer.Count<MeshletRecord>() || !buffers.ActiveMeshlets.Contains(owner.MeshletRoot,id)) {
            throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        }
        const auto &record = buffers.Meshlets.Get({id,1u})[0];
        const bool refined = record.RefinedGroup != InvalidOffset;
        if (record.Topology > 2u || !record.VertexCount || record.VertexCount > 64u || !record.TriangleCount ||
            record.TriangleCount > (record.Topology ? 16u : 48u) ||
            (!refined && uint64_t(record.TriangleOffset)+record.TriangleCount > buffers.MeshletTriangleIds.Buffer.Count<uint32_t>()) ||
            uint64_t(record.VertexOffset)+record.VertexCount > buffers.MeshletVertexCorners.Buffer.Count<uint32_t>() ||
            (!record.Topology && uint64_t(record.LocalTriangleOffset)+record.TriangleCount*3u > buffers.MeshletLocalTriangles.Buffer.Count<uint8_t>()))
            throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        finest += !refined;
    }
    if (finest > owner.Level0Count) throw std::logic_error("Meshlet retirement exceeds finest membership.");
    std::vector<uint32_t> sorted(clusters.begin(),clusters.end());
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end()) throw std::invalid_argument("Meshlet retirement repeats cluster identities.");
    MeshletIndexEdit edit{.Root=owner.MeshletRoot,.Removed=sorted};
    buffers.ActiveMeshlets.Update(std::span{&edit,1u});
    buffers.ReleaseMeshletStorage(sorted);
    owner.Level0Count -= finest;
    ++owner.MeshletRevision;
    std::vector<uint32_t> blocks;
    for (const auto id : sorted) if (blocks.empty() || blocks.back() != id/256u) blocks.push_back(id/256u);
    buffers.PosedMeshletBounds.UpdateBlocks(owner.StoreId,owner.MeshletRevision,blocks,
        [&](uint32_t block) { return buffers.ActiveMeshlets.HasBlock(owner.MeshletRoot,block); },owner.RenderTopology);
}
