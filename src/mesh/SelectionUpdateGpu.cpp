#include "mesh/MeshStore.h"

#include "Profile.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

namespace {
constexpr std::array Elements{Element::Vertex, Element::Edge, Element::Face};
constexpr std::array Domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
} // namespace

void MeshStore::UpdateSelection(state::Scene &r, mtl::ComputeChain &chain, std::span<const SelectionUpdate> updates) {
    if (updates.empty()) return;
    const profile::CpuScope scope{"UpdateSelection"};
    const auto &pipelines = GetMeshPipelines(r);
    const std::array block_arenas{&Buffers.Vertices.Blocks, &Buffers.EdgeHalfedges.Blocks, &Buffers.FaceTriangles.Blocks};
    // The workspace starts with the dirty-block bits, clear in a fresh workspace.
    const auto dirty_words = 3u * ((std::max({block_arenas[0]->HighWaterMark(), block_arenas[1]->HighWaterMark(), block_arenas[2]->HighWaterMark()}) + 31u) / 32u);
    // Every mesh update shares one mark, one block update and one root reduction.
    std::vector<SelectionMeshUpdate> meshes;
    meshes.reserve(updates.size());
    uint64_t seed_count = 0u, entry_bound = 0u;
    for (const auto &update : updates) {
        const auto &record = Records.at(update.StoreId);
        auto &mesh = meshes.emplace_back(SelectionMeshUpdate{
            .Connectivity = GetConnectivityRef(update.StoreId),
            .Owners = {record.Vertices.Index, record.EdgeData.Index, record.FaceData.Index},
            .FaceCount = Buffers.FaceTriangles.Count(record.FaceData),
            .Source = update.Source == Element::None ? InvalidOffset : uint32_t(std::ranges::find(Elements, update.Source) - Elements.begin()),
            .Root = 3u * update.StoreId,
        });
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto list = GetBlockList(update.StoreId, Domains[d]);
            mesh.Lists[d] = list.Gpu;
            mesh.Counts[d] = uint32_t(list.Blocks.size());
        }
        seed_count += update.Seeds.size();
        // The listed blocks enter the dirty list first, and the GPU marks append their incident blocks.
        // Listed blocks can have left the mesh, so the list holds them beside every block the mesh keeps.
        entry_bound += mesh.Counts[0] + mesh.Counts[1] + mesh.Counts[2] + update.Blocks[0].size() + update.Blocks[1].size() + update.Blocks[2].size();
    }
    constexpr uint32_t MeshWords = sizeof(SelectionMeshUpdate) / sizeof(uint32_t);
    const Range mesh_range{dirty_words, uint32_t(meshes.size() * MeshWords)};
    const Range seed_range{mesh_range.Offset + mesh_range.Count, uint32_t(seed_count * 3u)};
    const Range seed_updates{seed_range.Offset + seed_range.Count, uint32_t(seed_count)};
    const Range list_range{seed_updates.Offset + seed_updates.Count, uint32_t(3u + 2u * entry_bound)};
    mtl::Buffer work{BufferContext(), uint64_t(list_range.Offset + list_range.Count) * sizeof(uint32_t), SlotType::Buffer, mtl::BufferLifetime::Workspace};
    {
        const auto words = work.SetCount<uint32_t>(list_range.Offset + list_range.Count);
        std::ranges::copy(std::as_bytes(std::span{meshes}), std::as_writable_bytes(words.subspan(mesh_range.Offset, mesh_range.Count)).begin());
        auto seeds = std::as_writable_bytes(words.subspan(seed_range.Offset, seed_range.Count)).begin();
        auto owners = words.subspan(seed_updates.Offset, seed_updates.Count).begin();
        const auto list = words.subspan(list_range.Offset, list_range.Count);
        const auto dirty = words.first(dirty_words);
        uint32_t count = 0u;
        for (uint32_t u = 0u; u < updates.size(); ++u) {
            const auto &update = updates[u];
            seeds = std::ranges::copy(std::as_bytes(std::span{update.Seeds}), seeds).out;
            owners = std::fill_n(owners, update.Seeds.size(), u);
            for (uint32_t d = 0u; d < 3u; ++d)
                for (const auto block : update.Blocks[d]) {
                    auto &word = dirty[SelectionDirtyWord(d, block)];
                    const auto bit = 1u << (block % 32u);
                    if (word & bit) continue;
                    word |= bit;
                    list[3u + 2u * count] = (d << 30u) | block;
                    list[4u + 2u * count++] = u;
                }
        }
        std::ranges::copy(std::array{count, 1u, 1u}, list.begin());
    }
    const SelectionUpdatePushConstants pc{
        .CornersSlot = Buffers.FaceCorners.Buffer.Slot,
        .VerticesSlot = Buffers.Vertices.Buffer.Slot,
        .EdgeSharpnessSlot = Buffers.EdgeSharpness.Buffer.Slot,
        .FaceSharpnessSlot = Buffers.FaceSharpness.Buffer.Slot,
        .Blocks = {Buffers.Vertices.Blocks.Buffer.Slot, Buffers.EdgeHalfedges.Blocks.Buffer.Slot,
                   Buffers.FaceTriangles.Blocks.Buffer.Slot, Buffers.FaceCorners.Blocks.Buffer.Slot},
        .Masks = {Buffers.VertexSelection.Buffer.Slot, Buffers.EdgeSelection.Buffer.Slot, Buffers.FaceSelection.Buffer.Slot},
        .Leaves = {Buffers.VertexAggregates.Buffer.Slot, Buffers.EdgeAggregates.Buffer.Slot, Buffers.FaceAggregates.Buffer.Slot},
        .DirtySlot = work.Slot,
        .RootsSlot = Buffers.SelectionRoots.Buffer.Slot,
        .WorkSlot = work.Slot,
        .Updates = mesh_range.Offset, .Seeds = seed_range.Offset, .SeedUpdates = seed_updates.Offset, .List = list_range.Offset,
        .SeedCount = uint32_t(seed_count),
    };
    {
        const profile::CpuScope mark_scope{"SelectionMark"};
        chain.Groups(pipelines[MeshPass::MarkSelectionNeighbors], pc, uint32_t((seed_count * 32u + 255u) / 256u));
    }
    // Derived mask words are Persistent, so their pages are captured before the GPU rewrites them.
    if (std::ranges::any_of(meshes, [](const SelectionMeshUpdate &mesh) { return mesh.Source != InvalidOffset; })) {
        chain.Submit();
        const auto list = work.GetSpan<uint32_t>(list_range);
        std::array<std::vector<uint32_t>, 3> derived;
        for (uint32_t i = 0u; i < list[0]; ++i) {
            const auto entry = list[3u + 2u * i], source = meshes[list[4u + 2u * i]].Source, d = entry >> 30u;
            if (source != InvalidOffset && d != source) derived[d].push_back(entry & ((1u << 30u) - 1u));
        }
        for (uint32_t d = 0u; d < 3u; ++d) CaptureSelectionBlocks(Elements[d], derived[d]);
    }
    const profile::CpuScope update_scope{"SelectionUpdateBlocks"};
    chain.Indirect(pipelines[MeshPass::UpdateSelectionBlocks], pc, work, uint64_t(list_range.Offset) * sizeof(uint32_t));
    chain.Groups(pipelines[MeshPass::ReduceSelectionRoots], pc, 3u * uint32_t(meshes.size()));
    chain.Retain(std::move(work));
}

void MeshStore::ReconcileSelection(state::Scene &r, mtl::ComputeChain &chain, std::span<const Change> changes) {
    std::vector<SelectionUpdate> updates;
    for (const auto &change : changes) {
        const auto id = change.StoreId;
        if (id >= Records.size() || !Records[id].Alive) continue;
        if (!Records[id].SelectionSummary.Count) {
            std::ranges::fill(Buffers.SelectionRoots.GetMutable({3u * id, 3u}), SelectionAggregate{});
            continue;
        }
        auto &update = updates.emplace_back(SelectionUpdate{.StoreId = id});
        std::ranges::copy(std::span{change.Blocks}.first(3u), update.Blocks.begin());
        // Edge sharpness reaches the flags of incident vertices, and halfedge links reach edges and vertices.
        for (const auto d : {1u, SelectionHalfedgeDomain})
            for (const auto block : change.Blocks[d])
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) update.Seeds.push_back({d, block * MeshElementBlockWords + w, ~0u});
    }
    UpdateSelection(r, chain, updates);
}

Range MeshStore::GatherSelectedElements(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, Element element, BufferArena<uint32_t> &output) const {
    const auto selection = GetSelectedElements(id, element);
    std::vector<uint32_t> blocks;
    const auto count = selection.ForEachBlock([&](uint32_t block, uint32_t before) {
        blocks.push_back(block);
        blocks.push_back(before);
    });
    const auto handles = output.Allocate(count);
    if (!count) return handles;
    const auto range = chain.Scratch.Allocate(std::span<const uint32_t>{blocks});
    const SelectionGatherPushConstants pc{
        .Blocks = {chain.Scratch.Buffer.Slot, range.Offset}, .Output = {output.Buffer.Slot, handles.Offset},
        .MaskSlot = GetSelectionSlot(element), .Count = uint32_t(blocks.size() / 2u),
    };
    chain.Groups(GetMeshPipelines(r)[MeshPass::GatherSelectedElements], pc, pc.Count);
    return handles;
}
