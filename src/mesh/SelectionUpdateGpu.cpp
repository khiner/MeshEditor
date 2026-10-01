#include "mesh/MeshStore.h"

#include "Profile.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

namespace {
constexpr std::array Elements{Element::Vertex, Element::Edge, Element::Face};
constexpr std::array Domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
} // namespace

void MeshStore::UpdateSelection(state::Scene &r, std::span<const SelectionUpdate> updates) {
    if (updates.empty()) return;
    const profile::CpuScope scope{"UpdateSelection"};
    const auto &pipelines = GetMeshPipelines(r);
    const std::array block_arenas{&Buffers.Vertices.Blocks, &Buffers.EdgeHalfedges.Blocks, &Buffers.FaceTriangles.Blocks};
    const std::array aggregates{&Buffers.VertexAggregates, &Buffers.EdgeAggregates, &Buffers.FaceAggregates};
    // Growth clears the new words, and each update clears the bits it sets.
    const auto dirty_words = 3u * ((std::max({block_arenas[0]->HighWaterMark(), block_arenas[1]->HighWaterMark(), block_arenas[2]->HighWaterMark()}) + 31u) / 32u);
    SelectionDirty.Mirror({0u, dirty_words});
    std::vector<SelectionReducePushConstants> reductions;
    uint64_t scratch = 0u;
    for (const auto &update : updates) {
        auto &reduce = reductions.emplace_back(SelectionReducePushConstants{
            .Leaves = {aggregates[0]->Buffer.Slot, aggregates[1]->Buffer.Slot, aggregates[2]->Buffer.Slot},
            .RootsSlot = Buffers.SelectionRoots.Buffer.Slot, .Root = 3u * update.StoreId,
        });
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto list = GetBlockList(update.StoreId, Domains[d]);
            reduce.Lists[d] = list.Gpu;
            reduce.Counts[d] = uint32_t(list.Blocks.size());
        }
        scratch += update.Seeds.size() * 3u + 3u + reduce.Counts[0] + reduce.Counts[1] + reduce.Counts[2] +
            update.Blocks[0].size() + update.Blocks[1].size() + update.Blocks[2].size();
    }
    // All scratch is reserved before any range is written, so no range moves while it is in use.
    SelectionWork.Reset();
    SelectionWork.ReserveAdditional(uint32_t(scratch));
    auto dirty = SelectionDirty.GetMutable({0u, dirty_words});
    std::vector<SelectionUpdatePushConstants> constants;
    std::vector<Range> seeds, lists;
    for (uint32_t u = 0u; u < updates.size(); ++u) {
        const auto &update = updates[u];
        const auto &record = Records.at(update.StoreId);
        seeds.push_back(SelectionWork.Allocate(uint32_t(update.Seeds.size() * 3u)));
        std::ranges::copy(std::as_bytes(std::span{update.Seeds}), std::as_writable_bytes(SelectionWork.GetMutable(seeds.back())).begin());
        // The listed blocks enter the dirty list first, and the GPU marks append their incident blocks.
        // Listed blocks can have left the mesh, so the list holds them beside every block the mesh keeps.
        lists.push_back(SelectionWork.Allocate(uint32_t(3u + reductions[u].Counts[0] + reductions[u].Counts[1] + reductions[u].Counts[2] +
            update.Blocks[0].size() + update.Blocks[1].size() + update.Blocks[2].size())));
        auto list = SelectionWork.GetMutable(lists.back());
        uint32_t count = 0u;
        for (uint32_t d = 0u; d < 3u; ++d)
            for (const auto block : update.Blocks[d]) {
                auto &word = dirty[SelectionDirtyWord(d, block)];
                const auto bit = 1u << (block % 32u);
                if (word & bit) continue;
                word |= bit;
                list[3u + count++] = (d << 30u) | block;
            }
        std::ranges::copy(std::array{count, 1u, 1u}, list.begin());
        const std::array owners{record.Vertices, record.EdgeData, record.FaceData};
        constants.push_back({
            .Connectivity = GetConnectivityRef(update.StoreId),
            .CornersSlot = Buffers.FaceCorners.Buffer.Slot,
            .VerticesSlot = Buffers.Vertices.Buffer.Slot,
            .EdgeSharpnessSlot = Buffers.EdgeSharpness.Buffer.Slot,
            .FaceSharpnessSlot = Buffers.FaceSharpness.Buffer.Slot,
            .Blocks = {Buffers.Vertices.Blocks.Buffer.Slot, Buffers.EdgeHalfedges.Blocks.Buffer.Slot,
                       Buffers.FaceTriangles.Blocks.Buffer.Slot, Buffers.FaceCorners.Blocks.Buffer.Slot},
            .Owners = {owners[0].Index, owners[1].Index, owners[2].Index},
            .Masks = {Buffers.VertexSelection.Buffer.Slot, Buffers.EdgeSelection.Buffer.Slot, Buffers.FaceSelection.Buffer.Slot},
            .Leaves = {aggregates[0]->Buffer.Slot, aggregates[1]->Buffer.Slot, aggregates[2]->Buffer.Slot},
            .DirtySlot = SelectionDirty.Buffer.Slot,
            .List = {SelectionWork.Buffer.Slot, lists.back().Offset},
            .FaceCount = Buffers.FaceTriangles.Count(record.FaceData),
            .Source = update.Source == Element::None ? InvalidOffset : uint32_t(std::ranges::find(Elements, update.Source) - Elements.begin()),
        });
    }
    mtl::ComputeChain chain{BufferContext()};
    {
        const profile::CpuScope mark_scope{"SelectionMark"};
        for (uint32_t u = 0u; u < updates.size(); ++u) {
            auto pc = constants[u];
            pc.Items = {SelectionWork.Buffer.Slot, seeds[u].Offset};
            pc.Count = uint32_t(updates[u].Seeds.size());
            chain.Groups(pipelines[MeshPass::MarkSelectionNeighbors], pc, uint32_t((uint64_t(pc.Count) * 32u + 255u) / 256u));
        }
    }
    // Derived mask words are Persistent, so their pages are captured before the GPU rewrites them.
    if (std::ranges::any_of(updates, [](const SelectionUpdate &update) { return update.Source != Element::None; })) {
        chain.Submit();
        for (uint32_t u = 0u; u < updates.size(); ++u) {
            if (constants[u].Source == InvalidOffset) continue;
            const auto list = SelectionWork.Get(lists[u]);
            std::array<std::vector<uint32_t>, 3> derived;
            for (const auto entry : list.subspan(3u, list[0])) derived[entry >> 30u].push_back(entry & ((1u << 30u) - 1u));
            for (uint32_t d = 0u; d < 3u; ++d)
                if (d != constants[u].Source) CaptureSelectionBlocks(Elements[d], derived[d]);
        }
    }
    const profile::CpuScope update_scope{"SelectionUpdateBlocks"};
    for (uint32_t u = 0u; u < updates.size(); ++u) {
        auto pc = constants[u];
        pc.Items = {SelectionWork.Buffer.Slot, lists[u].Offset + 3u};
        chain.Indirect(pipelines[MeshPass::UpdateSelectionBlocks], pc, SelectionWork.Buffer, uint64_t(lists[u].Offset) * sizeof(uint32_t));
    }
    for (const auto &reduce : reductions) chain.Groups(pipelines[MeshPass::ReduceSelectionRoots], reduce, 3u);
    chain.Submit();
}

void MeshStore::ReconcileSelection(state::Scene &r, std::span<const Change> changes) {
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
    UpdateSelection(r, updates);
}

void MeshStore::GatherSelectedElements(state::Scene &r, uint32_t id, Element element, mtl::Buffer &output) const {
    const auto selection = GetSelectedElements(id, element);
    std::vector<uint32_t> blocks;
    const auto count = selection.ForEachBlock([&](uint32_t block, uint32_t before) {
        blocks.push_back(block);
        blocks.push_back(before);
    });
    output.SetUsedSize(uint64_t(count) * sizeof(uint32_t));
    output.CaptureWrite(0, output.UsedSize);
    if (!count) return;
    mtl::ComputeChain chain{BufferContext()};
    const auto range = chain.Scratch.Allocate(uint32_t(blocks.size()));
    std::ranges::copy(blocks, chain.Scratch.GetMutable(range).begin());
    const SelectionGatherPushConstants pc{
        .Blocks = {chain.Scratch.Buffer.Slot, range.Offset}, .Output = {output.Slot, 0u},
        .MaskSlot = GetSelectionSlot(element), .Count = uint32_t(blocks.size() / 2u),
    };
    chain.Groups(GetMeshPipelines(r)[MeshPass::GatherSelectedElements], pc, pc.Count);
    chain.Submit();
}
