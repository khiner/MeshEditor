#include "mesh/NormalDeriveGpu.h"

#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

// Input fields before assigning position and normal destinations.
// Empty for meshes without faces.
std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    if (meshes.Arenas().FaceTriangles.Count(record.FaceData) == 0) return {};
    return NormalDeriveEntry{
        .VerticesSlot = meshes.Slots().Vertices,
        .CornersSlot = meshes.Arenas().FaceCorners.Buffer.Slot,
        .VertexCount = meshes.Arenas().Vertices.Count(record.Vertices),
        .VertexBlocksSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .FaceCount = meshes.Arenas().FaceTriangles.Count(record.FaceData),
        .FaceBlocksSlot = meshes.Arenas().FaceTriangles.Blocks.Buffer.Slot,
        .CornerSectors = meshes.Arenas().CornerSectors.Ref(record.CornerSectors),
        .NormalSectors = meshes.Arenas().NormalSectors.Ref(record.NormalSectors),
    };
}

void EncodeDeriveNormals(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids, std::span<const NormalDeriveEntry> entries, NormalDerivePushConstants pc) {
    if (entries.empty()) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<uvec2> tiles;
    const auto append = [&](uint32_t i, bool faces) {
        const auto &entry = entries[i];
        const auto work = faces ? entry.FacesWork : entry.VerticesWork;
        if (work.Storage.Slot != InvalidSlot) {
            const auto count = faces ? entry.FaceWorkCount : entry.VertexWorkCount;
            for (uint32_t t = 0u; t < TileCount(count, TileElements); ++t) tiles.emplace_back(i, t);
        } else if (faces ? entry.FaceCount : entry.VertexCount) {
            const auto domain = faces ? MeshStore::ElementDomain::Face : MeshStore::ElementDomain::Vertex;
            WithDomain(meshes.Arenas(), domain, [&](const auto &arena) {
                arena.ForEachBlock(DomainSet(meshes.Get(ids[i]), domain), [&](uint32_t b, const auto &block) { if (block.Count) tiles.emplace_back(i, b); });
            });
        }
    };
    for (uint32_t i = 0u; i < entries.size(); ++i) append(i, true);
    const auto face_tiles = uint32_t(tiles.size());
    for (uint32_t i = 0u; i < entries.size(); ++i) append(i, false);
    mtl::Buffer jobs{chain.Buffers, std::as_bytes(entries), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    mtl::Buffer tile_buffer{chain.Buffers, BytesOf(tiles), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    pc.EntriesSlot = jobs.Slot;
    pc.TileMapSlot = tile_buffer.Slot;
    pc.Work = {};
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::VertexNormalDerive];
    for (uint32_t phase = 0; phase < 2; ++phase) {
        pc.Phase = phase;
        pc.FirstTile = phase == 0 ? 0u : face_tiles;
        chain.Groups(pipeline, pc, phase == 0 ? face_tiles : uint32_t(tiles.size()) - face_tiles, TileElements);
    }
    chain.Retain(std::move(jobs));
    chain.Retain(std::move(tile_buffer));
}

void CaptureNormalWrites(state::Registry &r, uint32_t id, const NormalDeriveEntry &entry, const BufferArena<uint32_t> &vertex_work, const BufferArena<uint32_t> &face_work) {
    if (!entry.VertexWorkCount && !entry.FaceWorkCount) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    if (!a.BaseVertexNormals.Buffer.History() && !a.BaseFaceNormals.Buffer.History() && !a.NormalSectors.Values.Buffer.History()) return;
    const auto &record = meshes.Get(id);
    const auto vertex_table = a.Vertices.Entries(record.Vertices), corner_table = a.FaceCorners.Entries(record.FaceCorners);
    // Sector normals live at root corners of the vertex fans.
    std::vector<uint32_t> corners;
    const auto incoming = a.VertexCorners.Buffer.GetSpan<uvec2>();
    const auto items = a.VertexFans.Items.Buffer.GetSpan<uvec2>();
    ForEachWorkHandle(vertex_work, entry.VerticesWork, entry.VertexWorkCount, 0u, [&](uint32_t v) {
        const auto fan = incoming[Resolve<MeshElementBlockSize>(vertex_table, ElementHandle{v}).Value()];
        for (auto item = fan.x; item < fan.x + fan.y; ++item) AddBlock(corners, Resolve<MeshElementBlockSize>(corner_table, ElementHandle{items[item].x}).Value());
    });
    PageFootprint pages;
    pages.Add(a.BaseVertexNormals.Buffer, WorkBlocks(vertex_work, entry.VerticesWork, entry.VertexWorkCount, vertex_table), BlockBytes<vec3>);
    pages.Add(a.BaseFaceNormals.Buffer, WorkBlocks(face_work, entry.FacesWork, entry.FaceWorkCount, a.FaceTriangles.Entries(record.FaceData)), BlockBytes<vec3>);
    pages.Layer(a.NormalSectors.Values, record.NormalSectors, a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>(), corners);
    pages.CaptureWrites();
}

namespace {
void EncodeDeriveBaseEntries(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids, std::span<const NormalDeriveEntry> entries, const BufferArena<uint32_t> &work) {
    const auto &meshes = r.Context.get<const MeshStore>();
    for (uint32_t i = 0u; i < entries.size(); ++i) CaptureNormalWrites(r, ids[i], entries[i], work, work);
    EncodeDeriveNormals(r, chain, ids, entries, {
                                                    .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
                                                    .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
                                                    .VertexNormalSlot = meshes.Slots().BaseVertexNormal,
                                                    .FaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                                    .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                                });
}
} // namespace

void EncodeDeriveMeshNormals(state::Registry &r, mtl::ComputeChain &chain, const BufferArena<uint32_t> &work, std::span<const LocalNormalWork> changes) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<uint32_t> ids;
    std::vector<NormalDeriveEntry> entries;
    for (const auto &change : changes) {
        auto entry = MakeDeriveEntryInputs(meshes, change.StoreId);
        if (!entry) continue;
        ids.push_back(change.StoreId);
        if (change.Vertices.Storage.Slot == InvalidSlot || change.Faces.Storage.Slot == InvalidSlot) throw std::invalid_argument("Local normal derivation requires canonical membership.");
        entry->VerticesWork = change.Vertices;
        entry->FacesWork = change.Faces;
        entry->VertexWorkCount = change.VertexCount;
        entry->FaceWorkCount = change.FaceCount;
        entries.push_back(*entry);
    }
    EncodeDeriveBaseEntries(r, chain, ids, entries, work);
}

AllNormalsWork EncodeAllNormalsMembership(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    const auto &meshes = r.Context.get<const MeshStore>();
    AllNormalsWork normals;
    std::vector<ElementWorkSeedJob> seeds;
    for (const auto id : ids) {
        const auto &record = meshes.Get(id);
        if (meshes.Arenas().FaceTriangles.Count(record.FaceData) == 0) continue;
        normals.Ids.push_back(id);
        seeds.push_back(PrepareElementMembershipWork(chain.Scratch, meshes.Arenas().Vertices, record.Vertices));
        seeds.push_back(PrepareElementMembershipWork(chain.Scratch, meshes.Arenas().FaceTriangles, record.FaceData));
        normals.Work.push_back(seeds[seeds.size() - 2u].Work);
        normals.Work.push_back(seeds.back().Work);
    }
    if (normals.Ids.empty()) return normals;
    EncodeElementMembershipWork(r, chain, seeds);
    EncodeSortElementWork(r, chain, normals.Work);
    return normals;
}

// Each entry is built here rather than with its membership work, since corner classification covers the sector records it references in between.
void EncodeDeriveAllNormals(state::Registry &r, mtl::ComputeChain &chain, const AllNormalsWork &normals) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<NormalDeriveEntry> entries;
    entries.reserve(normals.Ids.size());
    for (uint32_t i = 0u; i < normals.Ids.size(); ++i) {
        auto entry = *MakeDeriveEntryInputs(meshes, normals.Ids[i]);
        const auto v = normals.Work[2u * i], f = normals.Work[2u * i + 1u];
        CheckElementWork(chain.Scratch, v);
        CheckElementWork(chain.Scratch, f);
        entry.VerticesWork = v;
        entry.FacesWork = f;
        entry.VertexWorkCount = entry.VertexCount;
        entry.FaceWorkCount = entry.FaceCount;
        entries.push_back(entry);
    }
    EncodeDeriveBaseEntries(r, chain, normals.Ids, entries, chain.Scratch);
}
