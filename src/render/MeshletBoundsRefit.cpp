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
#include <stdexcept>
#include <vector>

void RefitCanonicalMeshletBounds(state::Scene &r,std::span<const MeshletBoundsRefitJob> jobs) {
    if (jobs.empty()) return;
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    std::vector<MeshletBoundsRefitPushConstants> dispatches;
    std::vector<std::pair<MeshBuffers *,std::vector<uint32_t>>> refitted;
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
        refitted.emplace_back(job.Owner,std::move(ids));
    }
    if (dispatches.empty()) return;
    mtl::ComputeChain chain{buffers.Ctx};
    const auto &pipeline=GetMeshPipelines(r)[MeshPass::MeshletBoundsRefit];
    for (const auto &pc:dispatches) chain.Groups(pipeline,pc,(pc.Count+31u)/32u,32u);
    chain.Submit();
    for (const auto &[owner,ids]:refitted) if (owner->SpatialRoot!=InvalidOffset) RefitMeshletSpatial(r,*owner,ids);
}

void StageDirtyPositionMeshlets(state::Scene &r,mtl::ComputeChain &chain,std::span<const state::Entity> entities) {
    auto &buffers=r.Context.get<GpuBuffers>();
    for (const auto entity:entities) {
        auto &owner=MeshBuffersOf(r,entity);
        const auto root=owner.PositionDirtyRoot;
        if (root==InvalidOffset) continue;
        const auto count=buffers.ActiveMeshlets.Count(root);
        if (count && buffers.ClusterGroupCount(owner)) {
            const profile::CpuScope scope{"PositionCoarseInvalidate"};
            std::vector<uint32_t> moved,groups;
            buffers.ActiveMeshlets.ForEach(root,[&](uint32_t meshlet) {
                if (!buffers.ActiveMeshlets.Contains(owner.MeshletRoot,meshlet)) return;
                moved.push_back(meshlet);
                if (const auto group=buffers.Meshlets.Get({meshlet,1u})[0].GroupIndex; group!=InvalidOffset) groups.push_back(group);
            });
            // The moved clusters refit their leaves along with every stale member.
            auto touched=InvalidateClusterGroups(r,entity,groups);
            touched.insert(touched.end(),moved.begin(),moved.end());
            std::ranges::sort(touched);
            touched.erase(std::unique(touched.begin(),touched.end()),touched.end());
            EditLodNodes(r,chain,owner,{},{},touched);
        }
        buffers.ActiveMeshlets.Release(root);
        owner.PositionDirtyRoot=InvalidOffset;
        buffers.PreludeStale=true;
    }
}
