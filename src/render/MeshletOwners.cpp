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
    const auto origin = topology == 0u ? 0u : meshes.RenderDomainFirst(owner, topology);
    auto &stored_origin = owner.ElementMeshletOrigins[topology];
    if (stored_origin != InvalidOffset && stored_origin != origin) {
        throw std::invalid_argument("Meshlet element origin changed without retiring its owners.");
    }
    auto &owners = render.ElementMeshlets[topology];
    const auto capacity = meshes.WithRenderDomain(owner, topology, [](const auto &arena, ElementSetRef) { return arena.Capacity(); }) / MeshElementBlockSize;
    if (std::ranges::any_of(blocks, [&](uint32_t block) { return block >= capacity; })) throw std::out_of_range("Meshlet owner block exceeds its element domain.");
    owners.ReserveBlocks(capacity);
    // A payload block belongs to the one mesh owning its element block, so an unbound block is new to this owner.
    owner.ElementMeshletBlockCounts[topology] += uint32_t(std::ranges::count_if(blocks, [&](uint32_t block) { return !owners.PayloadBlock(block); }));
    owners.Attach(blocks, InvalidOffset);
    std::vector<Range> payloads;
    payloads.reserve(blocks.size());
    for (const auto block : blocks) payloads.push_back(owners.Payload(block * MeshElementBlockSize, MeshElementBlockSize));
    owners.Values.Buffer.CaptureWriteRanges(payloads, sizeof(uint32_t));
    stored_origin = origin;
    MeshletOwnersPushConstants pc{
        .MeshletSlot = render.Meshlets.Buffer.Slot,
        .TriangleIdsSlot = render.MeshletTriangleIds.Buffer.Slot,
        .Topology = topology,
        .ElementOrigin = origin,
        .Owners = owners.Ref(),
        .BlockCount = capacity,
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

void RetireMeshletOwners(state::Scene &r, MeshStore::Record &owner, std::span<const uint32_t> clusters) {
    if (clusters.empty()) return;
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    std::vector<uint32_t> retired(clusters.begin(), clusters.end());
    std::array<std::vector<uint32_t>, 3> elements;
    std::ranges::sort(retired);
    for (const auto cluster : retired) {
        const auto &record = render.Meshlets.Get({cluster, 1u})[0];
        if (record.Topology >= 3u || record.RefinedGroup != InvalidOffset) {
            throw std::invalid_argument("Meshlet owner retirement requires finest canonical clusters.");
        }
        const auto origin = owner.ElementMeshletOrigins[record.Topology];
        if (origin == InvalidOffset) continue;
        for (const auto id : render.MeshletTriangleIds.Get({record.TriangleOffset, record.TriangleCount}))
            elements[record.Topology].push_back(origin + id);
    }
    for (uint32_t topology = 0u; topology < 3u; ++topology) {
        auto &handles = elements[topology];
        auto &owners = render.ElementMeshlets[topology];
        auto &blocks = owner.ElementMeshletBlockCounts[topology];
        std::ranges::sort(handles);
        for (size_t i = 0u; i < handles.size();) {
            const auto block = handles[i] / MeshElementBlockSize;
            auto end = i;
            while (end < handles.size() && handles[end] / MeshElementBlockSize == block) ++end;
            if (const auto payload = owners.PayloadBlock(block)) {
                // Only entries still naming a retired cluster clear, so replacement owners stay intact.
                auto &values = owners.Values.GetMutable({payload - 1u, 1u})[0];
                for (; i < end; ++i)
                    if (auto &value = values[handles[i] % MeshElementBlockSize]; std::ranges::binary_search(retired, value)) value = InvalidOffset;
                if (std::ranges::any_of(values, [](uint32_t value) { return value != InvalidOffset; })) continue;
                owners.Release(block);
                if (!blocks) throw std::logic_error("Meshlet owner block count underflow.");
                --blocks;
            }
            i = end;
        }
        if (!blocks) owner.ElementMeshletOrigins[topology] = InvalidOffset;
    }
}
