#include "mesh/NormalDeriveGpu.h"

#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    if (meshes.Arenas().FaceTriangles.Count(record.FaceData) == 0) return {};
    return NormalDeriveEntry{
        .VerticesSlot = meshes.Slots().Vertices,
        .VertexBlocksSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .FaceBlocksSlot = meshes.Arenas().FaceTriangles.Blocks.Buffer.Slot,
        .CornerSectors = meshes.Arenas().CornerSectors.Ref(record.CornerSectors),
        .NormalSectors = meshes.Arenas().NormalSectors.Ref(record.NormalSectors),
    };
}

void EncodeDeriveNormals(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids, std::span<const NormalDeriveEntry> entries, NormalDerivePushConstants pc) {
    if (entries.empty()) return;
    const auto &meshes = r.Resources.get<const MeshStore>();
    std::vector<uvec2> tiles;
    const auto append = [&](uint32_t i, bool faces) {
        const auto &entry = entries[i];
        const auto work = faces ? entry.FacesWork : entry.VerticesWork;
        if (work.Storage.Slot != InvalidSlot) {
            const auto count = faces ? entry.FaceWorkCount : entry.VertexWorkCount;
            for (uint32_t t = 0u; t < TileCount(count, TileElements); ++t) tiles.emplace_back(i, t);
        } else if (const auto &record = meshes.Get(ids[i]); faces ? meshes.Arenas().FaceTriangles.Count(record.FaceData) : meshes.Arenas().Vertices.Count(record.Vertices)) {
            const auto domain = faces ? MeshStore::ElementDomain::Face : MeshStore::ElementDomain::Vertex;
            WithDomain(meshes.Arenas(), domain, [&](const auto &arena) {
                arena.ForEachBlock(DomainSet(record, domain), [&](uint32_t b, const auto &block) { if (block.Count) tiles.emplace_back(i, b); });
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

void CaptureNormalWrites(state::Registry &r, uint32_t id, const NormalDeriveEntry &entry, const BufferArena<uint32_t> &work) {
    if (!entry.VertexWorkCount && !entry.FaceWorkCount) return;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    if (!a.BaseVertexNormals.Buffer.History() && !a.BaseFaceNormals.Buffer.History() && !a.NormalSectors.Values.Buffer.History()) return;
    const auto &record = meshes.Get(id);
    const auto vertex_table = a.Vertices.Entries(record.Vertices), corner_table = a.FaceCorners.Entries(record.FaceCorners);
    std::vector<uint32_t> corners;
    const auto incoming = a.VertexCorners.Buffer.GetSpan<uvec2>();
    const auto items = a.VertexFans.Items.Buffer.GetSpan<uvec2>();
    ForEachWorkHandle(work, entry.VerticesWork, entry.VertexWorkCount, 0u, [&](uint32_t v) {
        const auto fan = incoming[Resolve<MeshElementBlockSize>(vertex_table, ElementHandle{v}).Value()];
        for (auto item = fan.x; item < fan.x + fan.y; ++item) AddBlock(corners, Resolve<MeshElementBlockSize>(corner_table, ElementHandle{items[item].x}).Value());
    });
    PageFootprint pages;
    pages.Add(a.BaseVertexNormals.Buffer, WorkBlocks(work, entry.VerticesWork, entry.VertexWorkCount, vertex_table), BlockBytes<vec3>);
    pages.Add(a.BaseFaceNormals.Buffer, WorkBlocks(work, entry.FacesWork, entry.FaceWorkCount, a.FaceTriangles.Entries(record.FaceData)), BlockBytes<vec3>);
    pages.Layer(a.NormalSectors.Values, record.NormalSectors, a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>(), corners);
    pages.CaptureWrites();
}

namespace {
void EncodeDeriveBaseEntries(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids, std::span<const NormalDeriveEntry> entries) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    for (uint32_t i = 0u; i < entries.size(); ++i) CaptureNormalWrites(r, ids[i], entries[i], chain.Scratch);
    EncodeDeriveNormals(r, chain, ids, entries, {
                                                    .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
                                                    .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
                                                    .VertexNormalSlot = meshes.Slots().BaseVertexNormal,
                                                    .FaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                                    .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                                });
}
} // namespace

void EncodeDeriveMeshNormals(state::Registry &r, mtl::ComputeChain &chain, std::span<const LocalNormalWork> changes) {
    const auto &meshes = r.Resources.get<const MeshStore>();
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
    EncodeDeriveBaseEntries(r, chain, ids, entries);
}

void EncodeDeriveRecordNormals(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> store_ids) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto blocks = [](std::span<const uint32_t> entries) { return entries | std::views::filter([](uint32_t block) { return block != InvalidOffset; }) | std::ranges::to<std::vector>(); };
    std::vector<uint32_t> ids;
    std::vector<NormalDeriveEntry> entries;
    PageFootprint pages;
    for (const auto id : store_ids) {
        // An entry without work derives over its record's element blocks.
        const auto entry = MakeDeriveEntryInputs(meshes, id);
        if (!entry) continue;
        ids.push_back(id);
        entries.push_back(*entry);
        const auto &record = meshes.Get(id);
        pages.Add(a.BaseVertexNormals.Buffer, blocks(a.Vertices.Entries(record.Vertices)), BlockBytes<vec3>);
        pages.Add(a.BaseFaceNormals.Buffer, blocks(a.FaceTriangles.Entries(record.FaceData)), BlockBytes<vec3>);
        pages.Layer(a.NormalSectors.Values, record.NormalSectors, a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>(), blocks(a.FaceCorners.Entries(record.FaceCorners)));
    }
    pages.CaptureWrites();
    EncodeDeriveBaseEntries(r, chain, ids, entries);
}
