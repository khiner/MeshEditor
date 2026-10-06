#include "mesh/NormalDeriveGpu.h"

#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"

// Input fields before assigning position and normal destinations. Empty for meshes without faces.
std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    if (meshes.Arenas().FaceTriangles.Count(record.FaceData) == 0) return {};
    return NormalDeriveEntry{
        .Vertices = {meshes.Slots().Vertices, meshes.Arenas().Vertices.First(record.Vertices)},
        .Corners = {meshes.Arenas().FaceCorners.Buffer.Slot, meshes.Arenas().FaceCorners.First(record.FaceCorners)},
        .VertexCount = meshes.Arenas().Vertices.Count(record.Vertices),
        .VertexBlocksSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .FaceDataOffset = meshes.Arenas().FaceTriangles.First(record.FaceData),
        .FaceCount = meshes.Arenas().FaceTriangles.Count(record.FaceData),
        .FaceBlocksSlot = meshes.Arenas().FaceTriangles.Blocks.Buffer.Slot,
        .HasSectors = uint32_t(meshes.Get(store_id).SectorBlockCount != 0u),
    };
}

void EncodeDeriveNormals(state::Scene &r, mtl::ComputeChain &chain, std::span<const NormalDeriveEntry> entries, NormalDerivePushConstants pc) {
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
            const auto &blocks = faces ? meshes.Arenas().FaceTriangles.Blocks : meshes.Arenas().Vertices.Blocks;
            for (auto b = (faces ? entry.FaceDataOffset : entry.Vertices.Offset) / 256u; b != InvalidOffset; b = blocks.Get({b, 1u})[0].Next)
                if (blocks.Get({b, 1u})[0].Count) tiles.emplace_back(i, b);
        }
    };
    for (uint32_t i = 0u; i < entries.size(); ++i) append(i, true);
    const auto face_tiles = uint32_t(tiles.size());
    for (uint32_t i = 0u; i < entries.size(); ++i) append(i, false);
    mtl::Buffer jobs{chain.Buffers, std::as_bytes(entries), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    mtl::Buffer tile_buffer{chain.Buffers, as_bytes(tiles), SlotType::Buffer, mtl::BufferLifetime::Workspace};
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

void CaptureNormalWrites(state::Scene &r, const NormalDeriveEntry &entry, const BufferArena<uint32_t> &vertex_work, const BufferArena<uint32_t> &face_work) {
    if (!entry.VertexWorkCount && !entry.FaceWorkCount) return;
    const auto &a = r.Context.get<const MeshStore>().Arenas();
    if (!a.BaseVertexNormals.Buffer.History() && !a.BaseFaceNormals.Buffer.History() && !a.NormalSectors.Values.Buffer.History()) return;
    // Sector normals live at root corners of the vertex fans.
    std::vector<uint32_t> corners;
    const auto incoming = a.VertexCorners.Buffer.GetSpan<uvec2>();
    const auto items = a.VertexFans.Items.Buffer.GetSpan<uvec2>();
    ForEachWorkHandle(vertex_work, entry.VerticesWork, entry.VertexWorkCount, 0u, [&](uint32_t v) {
        if (v >= incoming.size()) return;
        for (auto item = incoming[v].x; item < incoming[v].x + incoming[v].y; ++item) AddBlock(corners, items[item].x);
    });
    PageFootprint pages;
    pages.Add(a.BaseVertexNormals.Buffer, WorkBlocks(vertex_work, entry.VerticesWork, entry.VertexWorkCount), BlockBytes<vec3>);
    pages.Add(a.BaseFaceNormals.Buffer, WorkBlocks(face_work, entry.FacesWork, entry.FaceWorkCount), BlockBytes<vec3>);
    pages.AttributeValues(a.NormalSectors, corners, false);
    pages.CaptureWrites();
}

namespace {
void EncodeDeriveBaseEntries(state::Scene &r, mtl::ComputeChain &chain, std::span<const NormalDeriveEntry> entries, const BufferArena<uint32_t> &work) {
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto &entry : entries) CaptureNormalWrites(r, entry, work, work);
    EncodeDeriveNormals(r, chain, entries, {
                                               .CornerSectors = meshes.Slots().CornerSector,
                                               .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
                                               .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
                                               .VertexNormalSlot = meshes.Slots().BaseVertexNormal,
                                               .NormalSectors = meshes.Slots().NormalSector,
                                               .FaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                               .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
                                           });
}
} // namespace

void EncodeDeriveMeshNormals(state::Scene &r, mtl::ComputeChain &chain, const BufferArena<uint32_t> &work, std::span<const LocalNormalWork> changes) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<NormalDeriveEntry> entries;
    for (const auto &change : changes) {
        auto entry = MakeDeriveEntryInputs(meshes, change.StoreId);
        if (!entry) continue;
        if (change.Vertices.Storage.Slot == InvalidSlot || change.Faces.Storage.Slot == InvalidSlot) throw std::invalid_argument("Local normal derivation requires canonical membership.");
        entry->VerticesWork = change.Vertices;
        entry->FacesWork = change.Faces;
        entry->VertexWorkCount = change.VertexCount;
        entry->FaceWorkCount = change.FaceCount;
        entries.push_back(*entry);
    }
    EncodeDeriveBaseEntries(r, chain, entries, work);
}

void EncodeDeriveAllNormals(state::Scene &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<ElementWork> work;
    std::vector<NormalDeriveEntry> entries;
    for (const auto id : ids) {
        auto entry = MakeDeriveEntryInputs(meshes, id);
        if (!entry) continue;
        const auto &record = meshes.Get(id);
        const auto v = PrepareElementMembershipWork(chain.Scratch, meshes.Arenas().Vertices, record.Vertices);
        const auto f = PrepareElementMembershipWork(chain.Scratch, meshes.Arenas().FaceTriangles, record.FaceData);
        seeds.push_back(v);
        seeds.push_back(f);
        work.push_back(v.Work);
        work.push_back(f.Work);
        entry->VerticesWork = v.Work;
        entry->FacesWork = f.Work;
        entry->VertexWorkCount = entry->VertexCount;
        entry->FaceWorkCount = entry->FaceCount;
        entries.push_back(*entry);
    }
    if (entries.empty()) return;
    EncodeElementMembershipWork(r, chain, seeds);
    EncodeSortElementWork(r, chain, work);
    chain.Submit();
    for (const auto domain : work) CheckElementWork(chain.Scratch, domain);
    EncodeDeriveBaseEntries(r, chain, entries, chain.Scratch);
}
