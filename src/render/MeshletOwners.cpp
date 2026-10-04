#include "render/MeshletOwners.h"
#include "gpu/MeshletOwnersPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"

namespace {
// Payload element IDs are relative to this canonical handle of the owner's element domain.
uint32_t ElementOrigin(const MeshStore &meshes, const MeshStore::Record &owner) {
    return owner.RenderTopology == 0u ? 0u : meshes.RenderDomainFirst(owner, owner.RenderTopology);
}
} // namespace

void PublishMeshletOwners(state::Scene &r, mtl::ComputeChain &chain, MeshStore::Record &owner, std::span<const Range> clusters, std::span<const uint32_t> blocks) {
    if (std::ranges::all_of(clusters,[](Range range) { return !range.Count; })) return;
    if (owner.RenderTopology >= 3u || owner.ExtrasFaces.Count) throw std::invalid_argument("Meshlet owners require a canonical render owner.");
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    const auto origin = ElementOrigin(meshes,owner);
    if (owner.ElementMeshletOrigin != InvalidOffset && owner.ElementMeshletOrigin != origin) {
        throw std::invalid_argument("Meshlet element origin changed without retiring its owners.");
    }
    auto &owners = render.ElementMeshlets[owner.RenderTopology];
    const auto capacity = meshes.WithRenderDomain(owner,owner.RenderTopology,[](const auto &arena, ElementSetRef) { return arena.Capacity(); })/MeshElementBlockSize;
    if (std::ranges::any_of(blocks,[&](uint32_t block) { return block >= capacity; })) throw std::out_of_range("Meshlet owner block exceeds its element domain.");
    owners.ReserveBlocks(capacity);
    // A payload block belongs to the one mesh owning its element block, so an unbound block is new to this owner.
    owner.ElementMeshletBlockCount += uint32_t(std::ranges::count_if(blocks,[&](uint32_t block) { return !owners.PayloadBlock(block); }));
    owners.Attach(blocks,InvalidOffset);
    std::vector<Range> payloads;
    payloads.reserve(blocks.size());
    for (const auto block : blocks) payloads.push_back(owners.Payload(block*MeshElementBlockSize,MeshElementBlockSize));
    owners.Values.Buffer.CaptureWriteRanges(payloads,sizeof(uint32_t));
    owner.ElementMeshletOrigin = origin;
    MeshletOwnersPushConstants pc{
        .MeshletSlot=render.Meshlets.Buffer.Slot,.TriangleIdsSlot=render.MeshletTriangleIds.Buffer.Slot,
        .Topology=owner.RenderTopology,.ElementOrigin=origin,.Owners=owners.Ref(),.BlockCount=capacity,
        .ErrorSlot=chain.Scratch.Buffer.Slot,
    };
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::MeshletOwners];
    // Each cluster names the owners of its own elements.
    chain.Concurrent([&] {
        for (const auto range : clusters) {
            pc.First = range.Offset;
            pc.Count = range.Count;
            chain.Groups(pipeline,pc,range.Count,64u);
        }
    });
}

void RetireMeshletOwners(state::Scene &r, MeshStore::Record &owner, std::span<const uint32_t> clusters) {
    if (clusters.empty() || owner.ElementMeshletOrigin == InvalidOffset) return;
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    auto &owners = render.ElementMeshlets[owner.RenderTopology];
    std::vector<uint32_t> retired(clusters.begin(),clusters.end()), elements;
    std::ranges::sort(retired);
    for (const auto cluster : retired) {
        const auto &record = render.Meshlets.Get({cluster,1u})[0];
        if (record.Topology != owner.RenderTopology || record.RefinedGroup != InvalidOffset) {
            throw std::invalid_argument("Meshlet owner retirement requires finest clusters of the owner's topology.");
        }
        for (const auto id : render.MeshletTriangleIds.Get({record.TriangleOffset,record.TriangleCount}))
            elements.push_back(owner.ElementMeshletOrigin+id);
    }
    std::ranges::sort(elements);
    for (size_t i = 0u; i < elements.size();) {
        const auto block = elements[i]/MeshElementBlockSize;
        auto end = i;
        while (end < elements.size() && elements[end]/MeshElementBlockSize == block) ++end;
        if (const auto payload = owners.PayloadBlock(block)) {
            // Only entries still naming a retired cluster clear, so replacement owners stay intact.
            auto &values = owners.Values.GetMutable({payload-1u,1u})[0];
            for (; i < end; ++i)
                if (auto &value = values[elements[i]%MeshElementBlockSize]; std::ranges::binary_search(retired,value)) value = InvalidOffset;
            if (std::ranges::any_of(values,[](uint32_t value) { return value != InvalidOffset; })) continue;
            owners.Release(block);
            if (!owner.ElementMeshletBlockCount) throw std::logic_error("Meshlet owner block count underflow.");
            --owner.ElementMeshletBlockCount;
        }
        i = end;
    }
    if (!owner.ElementMeshletBlockCount) owner.ElementMeshletOrigin = InvalidOffset;
}
