#include "render/MeshletStorage.h"
#include "render/GpuBufferOps.h"
#include "state/Registry.h"

void RetireMeshletStorage(state::Registry &r, MeshStore::Record &owner, std::span<const uint32_t> clusters) {
    if (clusters.empty()) return;
    if (!owner.MeshletData) throw std::invalid_argument("Meshlet retirement requires owned canonical membership.");
    auto &meshes = r.Context.get<MeshStore>();
    const auto &render = meshes.Render();
    uint32_t finest = 0u;
    for (const auto id : clusters) {
        if (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{id})) throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        const auto &record = render.Meshlets.Get({render.Meshlets.Address(owner.MeshletData, ElementHandle{id}).Value(), 1u})[0];
        const bool refined = record.RefinedGroup != InvalidOffset;
        if (record.Topology > 2u || !record.VertexCount || record.VertexCount > 64u || !record.TriangleCount ||
            record.TriangleCount > (record.Topology ? 16u : 48u) ||
            (!refined && uint64_t(record.TriangleOffset) + record.TriangleCount > owner.MeshletTriangles.Values.Count) ||
            uint64_t(record.VertexOffset) + record.VertexCount > owner.MeshletVertices.Values.Count ||
            (!record.Topology && uint64_t(record.LocalTriangleOffset) + record.TriangleCount > owner.MeshletLocalTriangles.Values.Count)) {
            throw std::invalid_argument("Meshlet retirement references foreign or invalid storage.");
        }
        finest += !refined;
    }
    if (finest > owner.Level0Count) throw std::logic_error("Meshlet retirement exceeds finest membership.");
    std::vector<uint32_t> sorted(clusters.begin(), clusters.end());
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end()) throw std::invalid_argument("Meshlet retirement repeats cluster identities.");
    // The posed bounds of the retired clusters' blocks update while their addresses still resolve.
    std::vector<uint32_t> addresses;
    addresses.reserve(sorted.size());
    for (const auto id : sorted) addresses.push_back(render.Meshlets.Address(owner.MeshletData, ElementHandle{id}).Value());
    std::ranges::sort(addresses);
    meshes.ReleaseMeshletStorage(owner, sorted);
    owner.Level0Count -= finest;
    ++owner.MeshletRevision;
    UpdatePosedMeshletBlocks(r, owner, addresses);
}
