#include "render/MeshletBoundsRefit.h"
#include "gpu/MeshletBoundsRefitPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "render/ClusterLodRepair.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/LodNodeEdit.h"
#include "render/MeshletIndex.h"
#include "render/MeshletSpatial.h"
#include "state/Scene.h"
#include "Profile.h"
#include "SortUnique.h"
#include <stdexcept>
#include <vector>

void RefitCanonicalMeshletBounds(state::Scene &r,mtl::ComputeChain &chain,std::span<const MeshletBoundsRefitJob> jobs) {
    if (jobs.empty()) return;
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    std::vector<MeshletBoundsRefitPushConstants> dispatches;
    std::vector<std::pair<uint32_t,std::vector<uint32_t>>> refitted;
    for (const auto &job:jobs) {
        if (!job.Storage || ElementWorkEmpty(*job.Storage,job.Meshlets)) continue;
        if (!job.Owner) throw std::invalid_argument("Meshlet bounds refit requires a render owner.");
        CheckElementWork(*job.Storage,job.Meshlets);
        std::vector<uint32_t> ids;
        ForEachWorkElement(*job.Storage,job.Meshlets,[&](uint32_t id) { ids.push_back(id); });
        buffers.Meshlets.Buffer.CaptureWriteElements(ids,sizeof(MeshletRecord));
        dispatches.push_back({.Work=job.Meshlets,.Count=uint32_t(ids.size()),
            .MeshletSlot=buffers.Meshlets.Buffer.Slot,.MeshletVertexSlot=buffers.MeshletVertexCorners.Buffer.Slot,
            .LocalTrianglesSlot=buffers.MeshletLocalTriangles.Buffer.Slot,.CornerSlot=meshes.Arenas().FaceCorners.Buffer.Slot,
            .VertexSlot=meshes.Slots().Vertices,.VertexOffset=job.Owner->Vertices.Offset});
        if (job.Owner->SpatialRoot!=InvalidOffset) refitted.emplace_back(job.Owner->StoreId,std::move(ids));
    }
    if (dispatches.empty()) return;
    const auto &pipeline=GetMeshPipelines(r)[MeshPass::MeshletBoundsRefit];
    // Each job refits its own owner's clusters.
    chain.Concurrent([&] { for (const auto &pc:dispatches) chain.Groups(pipeline,pc,(pc.Count+31u)/32u,32u); });
    if (refitted.empty()) return;
    chain.AfterSubmit([&r,refitted=std::move(refitted)] {
        auto &buffers=r.Context.get<GpuBuffers>();
        for (const auto &[store_id,ids]:refitted)
            if (auto *owner=buffers.TryMeshOf(store_id); owner && owner->SpatialRoot!=InvalidOffset) RefitMeshletSpatial(r,*owner,ids);
    });
}

void StageDirtyPositionMeshlets(state::Scene &r,mtl::ComputeChain &chain,std::span<const state::Entity> entities) {
    const profile::CpuScope scope{"PositionCoarseInvalidate"};
    auto &buffers=r.Context.get<GpuBuffers>();
    // Each coarse owner's moved clusters, whose groups seed its stale closure.
    std::vector<ClusterGroupSeeds> seeds;
    std::vector<std::vector<uint32_t>> moved;
    for (const auto entity:entities) {
        auto &owner=MeshBuffersOf(r,entity);
        const auto root=owner.PositionDirtyRoot;
        if (root==InvalidOffset) continue;
        if (buffers.ActiveMeshlets.Count(root) && buffers.ClusterGroupCount(owner)) {
            auto &groups=seeds.emplace_back(ClusterGroupSeeds{.Entity=entity}).Groups;
            auto &owner_moved=moved.emplace_back();
            buffers.ActiveMeshlets.ForEach(root,[&](uint32_t meshlet) {
                if (!buffers.ActiveMeshlets.Contains(owner.MeshletRoot,meshlet)) return;
                owner_moved.push_back(meshlet);
                if (const auto group=buffers.Meshlets.Get({meshlet,1u})[0].GroupIndex; group!=InvalidOffset) groups.push_back(group);
            });
        }
        buffers.ActiveMeshlets.Release(root);
        owner.PositionDirtyRoot=InvalidOffset;
        buffers.PreludeStale=true;
    }
    auto touched=InvalidateClusterGroups(r,seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i=0u;i<seeds.size();++i) {
        // The moved clusters refit their leaves along with every stale member.
        touched[i].insert(touched[i].end(),moved[i].begin(),moved[i].end());
        SortUnique(touched[i]);
        refits.push_back(EditLodNodes(r,chain,MeshBuffersOf(r,seeds[i].Entity),{},{},touched[i]));
    }
    RecordLodNodeRefits(r,chain,refits);
}
