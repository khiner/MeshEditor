#include "render/MeshletOwners.h"
#include "gpu/MeshletOwnersPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"

void PublishMeshletOwners(state::Scene &r, mtl::ComputeChain &chain, MeshStore::Record &owner, uint32_t topology, std::span<const Range> clusters, std::span<const uint32_t> blocks) {
    if (std::ranges::all_of(clusters, [](Range range) { return !range.Count; })) return;
    if (topology >= 3u || owner.ExtrasFaces.Count) throw std::invalid_argument("Meshlet owners require a canonical render owner.");
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    auto &owners = render.ElementMeshlets[topology];
    auto &layer = owner.ElementMeshlets[topology];
    const auto bound = meshes.WithRenderDomain(owner, topology, [&](const auto &arena, ElementSetRef set) {
        const auto entries = arena.Entries(set);
        for (const auto local : blocks)
            if (local >= entries.size() || entries[local] == InvalidOffset) throw std::out_of_range("Meshlet owner block is outside its element domain.");
        return uint32_t(entries.size());
    });
    // The record's first owners start every element without one, and a block gains its first owner unless an entry there already names a cluster.
    const bool fresh = !layer.Count;
    if (fresh) {
        owners.Cover(layer, bound);
        owners.Clear(layer, {0u, bound});
    }
    const auto values = owners.View(layer).Values;
    std::vector<uint32_t> written;
    written.reserve(blocks.size());
    for (const auto local : blocks) {
        if (fresh || std::ranges::all_of(values.subspan(uint64_t(local) * MeshElementBlockSize, MeshElementBlockSize), [](uint32_t value) { return value == InvalidOffset; })) ++owner.ElementMeshletBlockCounts[topology];
        written.push_back(layer.Offset + local);
    }
    owners.Values.Buffer.CaptureWriteElements(written, sizeof(std::remove_cvref_t<decltype(owners)>::Block));
    MeshletOwnersPushConstants pc{
        .Meshlets = render.Meshlets.Slots(),
        .MeshletSet = owner.MeshletData.Index,
        .MeshletSlot = render.Meshlets.Buffer.Slot,
        .TriangleIdsSlot = render.MeshletTriangleIds.Buffer.Slot,
        .TriangleIdBase = owner.MeshletTriangles.Values.Offset,
        .Topology = topology,
        .Owners = owners.Ref(layer),
        .HandleBound = bound * MeshElementBlockSize,
        .ErrorSlot = chain.Scratch.Buffer.Slot,
    };
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::MeshletOwners];
    // Each cluster names the owners of its own elements.
    chain.Concurrent([&] {
        for (const auto range : clusters) {
            pc.First = range.Offset;
            pc.Count = range.Count;
            chain.Groups(pipeline, pc, range.Count, 64u);
        }
    });
}

void RetireMeshletOwners(state::Scene &r, MeshStore::Record &owner, std::span<const uint32_t> clusters, std::span<const uint32_t> retired) {
    if (clusters.empty()) return;
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    std::vector<uint32_t> sorted(clusters.begin(), clusters.end());
    std::array<std::vector<uint32_t>, 3> elements;
    uint32_t topologies = 0u;
    std::ranges::sort(sorted);
    for (const auto cluster : sorted) {
        const auto &record = render.Meshlets.Get({render.Meshlets.Address(owner.MeshletData, ElementHandle{cluster}).Value(), 1u})[0];
        if (record.Topology >= 3u || record.RefinedGroup != InvalidOffset) {
            throw std::invalid_argument("Meshlet owner retirement requires finest canonical clusters.");
        }
        if (!owner.ElementMeshletBlockCounts[record.Topology]) continue;
        topologies |= 1u << record.Topology;
        elements[record.Topology].append_range(render.MeshletTriangleIds.Get(owner.MeshletTriangles.Values).subspan(record.TriangleOffset, record.TriangleCount));
    }
    for (uint32_t topology = 0u; topology < 3u; ++topology) {
        auto &handles = elements[topology];
        if (topologies >> topology & 1u) handles.append_range(retired);
        auto &owners = render.ElementMeshlets[topology];
        auto &range = owner.ElementMeshlets[topology];
        auto &blocks = owner.ElementMeshletBlockCounts[topology];
        std::ranges::sort(handles);
        for (size_t i = 0u; i < handles.size();) {
            const auto block = handles[i] / MeshElementBlockSize;
            auto end = i;
            while (end < handles.size() && handles[end] / MeshElementBlockSize == block) ++end;
            if (block >= range.Count) {
                i = end;
                continue;
            }
            // Only entries still naming a retired cluster clear, so replacement owners stay intact, and a block left without an owner leaves the count.
            const auto values = owners.Edit(range, {block * MeshElementBlockSize, MeshElementBlockSize});
            const auto owned = [&] { return std::ranges::any_of(values, [](uint32_t value) { return value != InvalidOffset; }); };
            const bool held = owned();
            for (; i < end; ++i)
                if (auto &value = values[handles[i] % MeshElementBlockSize]; std::ranges::binary_search(sorted, value)) value = InvalidOffset;
            if (!held || owned()) continue;
            if (!blocks) throw std::logic_error("Meshlet owner block count underflow.");
            --blocks;
        }
        if (!blocks) owners.Release(range);
    }
}
