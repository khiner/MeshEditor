#include "render/MeshletBoundsRefit.h"
#include "Profile.h"
#include "SortUnique.h"
#include "gpu/MeshletBoundsRefitPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "render/ClusterLodRepair.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/LodNodeEdit.h"
#include "render/MeshletSpatial.h"
#include "state/Scene.h"
#include <stdexcept>
#include <vector>

void RefitCanonicalMeshletBounds(state::Scene &r, mtl::ComputeChain &chain, std::span<const MeshletBoundsRefitJob> jobs) {
    if (jobs.empty()) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    auto &render = meshes.Render();
    std::vector<MeshletBoundsRefitPushConstants> dispatches;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> refitted;
    for (const auto &job : jobs) {
        if (!job.Storage || ElementWorkEmpty(*job.Storage, job.Meshlets)) continue;
        if (!job.Owner) throw std::invalid_argument("Meshlet bounds refit requires a render owner.");
        CheckElementWork(*job.Storage, job.Meshlets);
        std::vector<uint32_t> addresses;
        ForEachWorkElement(*job.Storage, job.Meshlets, [&](uint32_t address) { addresses.push_back(address); });
        render.Meshlets.Buffer.CaptureWriteElements(addresses, sizeof(MeshletRecord));
        dispatches.push_back({.MeshletAddresses = job.Meshlets, .Count = uint32_t(addresses.size()), .MeshletSlot = render.Meshlets.Buffer.Slot, .MeshletVertexSlot = render.MeshletVertexCorners.Buffer.Slot, .LocalTrianglesSlot = render.MeshletLocalTriangles.Buffer.Slot, .CornerSlot = meshes.Arenas().FaceCorners.Buffer.Slot, .VertexSlot = meshes.Slots().Vertices, .VertexBase = job.Owner->MeshletVertices.Values.Offset, .LocalTriangleBase = job.Owner->MeshletLocalTriangles.Values.Offset, .VertexTable = meshes.Arenas().Vertices.TableRef(job.Owner->Vertices), .CornerTable = meshes.Arenas().FaceCorners.TableRef(job.Owner->FaceCorners)});
        // The spatial tree refits the finest triangle clusters at their handles.
        if (job.Owner->SpatialRoot != InvalidOffset) {
            std::vector<uint32_t> handles;
            for (const auto address : addresses) {
                const auto &cluster = render.Meshlets.Get({address, 1u})[0];
                if (cluster.Topology == 0u && cluster.RefinedGroup == InvalidOffset) handles.push_back(render.Meshlets.Handle(ElementAddress::Stored(address)).Value);
            }
            if (!handles.empty()) refitted.emplace_back(job.Owner->StoreId, std::move(handles));
        }
    }
    if (dispatches.empty()) return;
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::MeshletBoundsRefit];
    // Each job refits its own owner's clusters.
    chain.Concurrent([&] { for (const auto &pc:dispatches) chain.Groups(pipeline,pc,(pc.Count+31u)/32u,32u); });
    if (refitted.empty()) return;
    chain.AfterSubmit([&r, refitted = std::move(refitted)] {
        auto &meshes = r.Context.get<MeshStore>();
        for (const auto &[store_id, ids] : refitted)
            if (const auto *owner = meshes.TryGet(store_id); owner && owner->SpatialRoot != InvalidOffset) RefitMeshletSpatial(r, *owner, ids);
    });
}

void StageDirtyPositionMeshlets(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> entities) {
    const profile::CpuScope scope{"PositionCoarseInvalidate"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    // Each coarse owner's moved clusters, whose groups seed its stale closure.
    std::vector<ClusterGroupSeeds> seeds;
    std::vector<std::vector<uint32_t>> moved;
    for (const auto entity : entities) {
        auto &owner = EditRecordOf(r, entity);
        if (!meshes.HasDirtyPositions(owner)) continue;
        // The owner's dirty clusters clear, and with coarse geometry their groups seed its stale closure.
        const bool coarse = meshes.ClusterGroupCount(owner) != 0u;
        auto &groups = seeds.emplace_back(ClusterGroupSeeds{.Entity = entity}).Groups;
        auto &owner_moved = moved.emplace_back();
        const auto entries = render.Meshlets.Entries(owner.MeshletData);
        for (const auto block : entries) {
            if (block == InvalidOffset) continue;
            const auto &members = render.Meshlets.Blocks.Get({block, 1u})[0];
            const auto dirty = render.PositionDirty.Get({block, 1u})[0] & members.Live[0];
            if (!dirty) continue;
            render.PositionDirty.GetMutable({block, 1u})[0] = 0u;
            if (!coarse) continue;
            for (auto bits = dirty; bits; bits &= bits - 1u) {
                const auto slot = uint32_t(std::countr_zero(bits));
                owner_moved.push_back(members.Local * RenderBlockSize + slot);
                if (const auto group = render.Meshlets.Get({block * RenderBlockSize + slot, 1u})[0].GroupIndex; group != InvalidOffset) groups.push_back(group);
            }
        }
        if (!coarse) {
            seeds.pop_back();
            moved.pop_back();
        }
        buffers.PreludeStale = true;
    }
    auto touched = InvalidateClusterGroups(r, seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < seeds.size(); ++i) {
        // The moved clusters refit their leaves along with every stale member.
        touched[i].insert(touched[i].end(), moved[i].begin(), moved[i].end());
        SortUnique(touched[i]);
        refits.push_back(EditLodNodes(r, chain, EditRecordOf(r, seeds[i].Entity), {}, {}, touched[i]));
    }
    RecordLodNodeRefits(r, chain, refits);
}
