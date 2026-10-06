#include "mesh/MeshStore.h"

#include "Profile.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

namespace {
constexpr std::array Elements{Element::Vertex, Element::Edge, Element::Face};
constexpr std::array Domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};

// Upper bound on the neighbor marks, read from connectivity metadata only.
// A vertex fan visits at most one face and two edges per incoming corner.
uint64_t MarkBound(const MeshStore &meshes, const MeshStore::SelectionUpdate &update) {
    const auto &a = meshes.Arenas();
    const std::array members{a.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(), a.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(), a.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>(), a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>()};
    const auto fans = a.VertexCorners.Buffer.GetSpan<uvec2>();
    const auto faces = a.FaceRanges.Buffer.GetSpan<uvec2>();
    uint64_t bound = 0u;
    for (const auto &seed : update.Seeds) {
        bound += seed.Domain != SelectionHalfedgeDomain;
        const auto block = seed.Word / MeshElementBlockWords;
        if (block >= members[seed.Domain].size()) continue;
        const auto live = seed.Bits & members[seed.Domain][block].Live[seed.Word % MeshElementBlockWords];
        if (seed.Domain == 1u || seed.Domain == SelectionHalfedgeDomain) {
            bound += uint64_t(std::popcount(live)) * (seed.Domain == 1u ? 4u : 3u);
        } else
            for (auto bits = live; bits; bits &= bits - 1u) {
                const auto handle = seed.Word * 32u + uint32_t(std::countr_zero(bits));
                bound += seed.Domain == 0u ? 3ull * fans[handle].y : 2ull * (faces[handle].y - faces[handle].x);
            }
    }
    return bound;
}
} // namespace

void MeshStore::UpdateSelection(state::Scene &r, mtl::ComputeChain &chain, std::span<const SelectionUpdate> updates) {
    if (updates.empty()) return;
    const profile::CpuScope scope{"UpdateSelection"};
    const auto &pipelines = GetMeshPipelines(r);
    // Every mesh update shares one mark and one block update.
    std::vector<SelectionMeshUpdate> meshes;
    meshes.reserve(updates.size());
    uint64_t seed_count = 0u, entry_bound = 0u;
    for (const auto &update : updates) {
        const auto &record = Records.at(update.StoreId);
        meshes.emplace_back(SelectionMeshUpdate{
            .Connectivity = GetConnectivityRef(update.StoreId),
            .Owners = {record.Vertices.Index, record.EdgeData.Index, record.FaceData.Index},
            .FaceCount = Buffers.FaceTriangles.Count(record.FaceData),
            .Source = update.Source == Element::None ? InvalidOffset : uint32_t(std::ranges::find(Elements, update.Source) - Elements.begin()),
            .Root = 3u * update.StoreId,
        });
        seed_count += update.Seeds.size();
        const uint64_t owned = (record.Vertices ? Buffers.Vertices.Set(record.Vertices).BlockCount : 0u) +
            uint64_t(record.EdgeData ? Buffers.EdgeHalfedges.Set(record.EdgeData).BlockCount : 0u) +
            (record.FaceData ? Buffers.FaceTriangles.Set(record.FaceData).BlockCount : 0u);
        // A seed's own block may already have left the mesh.
        entry_bound += std::min(owned + update.Seeds.size(), MarkBound(*this, update)) + update.Blocks[0].size() + update.Blocks[1].size() + update.Blocks[2].size();
    }
    const auto dirty_words = std::bit_ceil(std::max(2ull, 2ull * entry_bound));
    constexpr uint32_t MeshWords = sizeof(SelectionMeshUpdate) / sizeof(uint32_t);
    if (dirty_words + meshes.size() * MeshWords + seed_count * 4u + 3u + 2u * entry_bound >= InvalidOffset)
        throw std::length_error("Selection update workspace exceeds canonical addressing.");
    const Range mesh_range{uint32_t(dirty_words), uint32_t(meshes.size() * MeshWords)};
    const Range seed_range{mesh_range.Offset + mesh_range.Count, uint32_t(seed_count * 3u)};
    const Range seed_updates{seed_range.Offset + seed_range.Count, uint32_t(seed_count)};
    const Range list_range{seed_updates.Offset + seed_updates.Count, uint32_t(3u + 2u * entry_bound)};
    auto work = std::make_shared<mtl::Buffer>(BufferContext(), uint64_t(list_range.Offset + list_range.Count) * sizeof(uint32_t), SlotType::Buffer, mtl::BufferLifetime::Workspace);
    profile::RecordCounter("SelectionWorkspaceBytes", uint64_t(list_range.Offset + list_range.Count) * sizeof(uint32_t));
    {
        const auto words = work->SetCount<uint32_t>(list_range.Offset + list_range.Count);
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
                    const auto key = ((d << 30u) | block) + 1u;
                    auto slot = SelectionDirtyHash(key, uint32_t(dirty_words - 1u));
                    while (dirty[slot] && dirty[slot] != key) slot = (slot + 1u) & uint32_t(dirty_words - 1u);
                    if (dirty[slot] == key) continue;
                    dirty[slot] = key;
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
        .Blocks = {Buffers.Vertices.Blocks.Buffer.Slot, Buffers.EdgeHalfedges.Blocks.Buffer.Slot, Buffers.FaceTriangles.Blocks.Buffer.Slot, Buffers.FaceCorners.Blocks.Buffer.Slot},
        .Masks = {Buffers.VertexSelection.Buffer.Slot, Buffers.EdgeSelection.Buffer.Slot, Buffers.FaceSelection.Buffer.Slot},
        .Leaves = {Buffers.VertexAggregates.Buffer.Slot, Buffers.EdgeAggregates.Buffer.Slot, Buffers.FaceAggregates.Buffer.Slot},
        .Hidden = {Buffers.VertexHidden.Buffer.Slot, Buffers.EdgeHidden.Buffer.Slot, Buffers.FaceHidden.Buffer.Slot},
        .DirtyMask = uint32_t(dirty_words - 1u),
        .WorkSlot = work->Slot,
        .Updates = mesh_range.Offset,
        .Seeds = seed_range.Offset,
        .SeedUpdates = seed_updates.Offset,
        .List = list_range.Offset,
        .SeedCount = uint32_t(seed_count),
    };
    {
        const profile::CpuScope mark_scope{"SelectionMark"};
        chain.Groups(pipelines[MeshPass::MarkSelectionNeighbors], pc, uint32_t((seed_count * 32u + 255u) / 256u));
    }
    // Derived mask words are Persistent, so their pages are captured before the GPU rewrites them.
    if (std::ranges::any_of(meshes, [](const SelectionMeshUpdate &mesh) { return mesh.Source != InvalidOffset; })) {
        chain.Submit();
        const auto list = work->GetSpan<uint32_t>(list_range);
        std::array<std::vector<uint32_t>, 3> derived;
        for (uint32_t i = 0u; i < list[0]; ++i) {
            const auto entry = list[3u + 2u * i], source = meshes[list[4u + 2u * i]].Source, d = entry >> 30u;
            if (source != InvalidOffset && d != source) derived[d].push_back(entry & ((1u << 30u) - 1u));
        }
        for (uint32_t d = 0u; d < 3u; ++d) CaptureSelectionBlocks(Elements[d], derived[d]);
    }
    const profile::CpuScope update_scope{"SelectionUpdateBlocks"};
    chain.Indirect(pipelines[MeshPass::UpdateSelectionBlocks], pc, *work, uint64_t(list_range.Offset) * sizeof(uint32_t));
    chain.AfterSubmit([this, work, list_range, meshes = std::move(meshes)] {
        const profile::CpuScope scope{"SelectionIndexUpdate"};
        const auto list = work->GetSpan<uint32_t>(list_range);
        profile::RecordCounter("SelectionUpdatedBlocks", list[0]);
        std::vector<std::array<std::vector<uint32_t>, 3>> blocks(meshes.size());
        for (uint32_t i = 0u; i < list[0]; ++i) {
            const auto entry = list[3u + 2u * i];
            blocks[list[4u + 2u * i]][entry >> 30u].push_back(entry & ((1u << 30u) - 1u));
        }
        const std::array members{Buffers.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(), Buffers.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(), Buffers.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>()};
        const std::array leaves{Buffers.VertexAggregates.Buffer.GetSpan<SelectionAggregate>(), Buffers.EdgeAggregates.Buffer.GetSpan<SelectionAggregate>(), Buffers.FaceAggregates.Buffer.GetSpan<SelectionAggregate>()};
        for (uint32_t i = 0u; i < meshes.size(); ++i)
            for (uint32_t d = 0u; d < 3u; ++d)
                Buffers.SelectionTree.Update(meshes[i].Root + d, blocks[i][d], meshes[i].Owners[d], members[d], leaves[d]);
    });
}

void MeshStore::ReconcileSelection(state::Scene &r, mtl::ComputeChain &chain, std::span<const Change> changes) {
    std::vector<SelectionUpdate> updates;
    for (const auto &change : changes) {
        const auto id = change.StoreId;
        if (id >= Records.size() || !Records[id].Alive || !Records[id].SelectionSummary.Count) {
            for (uint32_t d = 0u; d < 3u; ++d) Buffers.SelectionTree.Release(3u * id + d);
            continue;
        }
        auto &update = updates.emplace_back(SelectionUpdate{.StoreId = id});
        std::ranges::copy(std::span{change.Blocks}.first(3u), update.Blocks.begin());
        // Position writes reach incident face bounds; edge sharpness and halfedge links reach neighboring flags.
        for (const auto d : {0u, 1u, SelectionHalfedgeDomain})
            for (const auto block : change.Blocks[d])
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) update.Seeds.push_back({d, block * MeshElementBlockWords + w, ~0u});
    }
    UpdateSelection(r, chain, updates);
}

void MeshStore::RebuildSelectionIndex(std::span<const uint32_t> ids) {
    const std::array members{Buffers.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(), Buffers.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(), Buffers.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>()};
    const std::array leaves{Buffers.VertexAggregates.Buffer.GetSpan<SelectionAggregate>(), Buffers.EdgeAggregates.Buffer.GetSpan<SelectionAggregate>(), Buffers.FaceAggregates.Buffer.GetSpan<SelectionAggregate>()};
    for (const auto id : ids)
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto owner = DomainSet(Records.at(id), Domains[d]).Index;
            Buffers.SelectionTree.Release(3u * id + d);
            Buffers.SelectionTree.Update(3u * id + d, GetBlockList(id, Domains[d]).Blocks, owner, members[d], leaves[d]);
        }
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
        .Blocks = {chain.Scratch.Buffer.Slot, range.Offset},
        .Output = {output.Buffer.Slot, handles.Offset},
        .MaskSlot = GetSelectionSlot(element),
        .Count = uint32_t(blocks.size() / 2u),
    };
    chain.Groups(GetMeshPipelines(r)[MeshPass::GatherSelectedElements], pc, pc.Count);
    return handles;
}
