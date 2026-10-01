#include "render/MeshletPatchWork.h"
#include "Profile.h"

#include "mesh/MeshStore.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/MeshletOwners.h"
#include "state/Scene.h"
#include <map>

MeshletPatch PlanMeshletPatch(state::Scene &r, const MeshBuffers &owner, const BufferArena<uint32_t> &storage, const MeshletPatchInput &input) {
    const profile::CpuScope scope{"MeshletPatchPlan"};
    if (input.Sources.size() != input.Added.Count || uint64_t(input.Added.Offset)+input.Added.Count > UINT32_MAX) {
        throw std::invalid_argument("Meshlet patch provenance must cover the emitted triangle run.");
    }
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &index = buffers.ActiveMeshlets;
    const auto owners = buffers.ElementMeshlets[0].View();
    const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto triangle_ids = buffers.MeshletTriangleIds.Buffer.GetSpan<uint32_t>();
    const auto contains = [&](ElementWork work, uint32_t element) {
        return work.Storage.Slot != InvalidSlot && ElementWorkRank(storage.Get(WorkStorageRange(work)),work,element) != InvalidOffset;
    };
    // The live finest cluster whose payload holds the triangle.
    const auto cluster_of = [&](uint32_t triangle) {
        const auto cluster = owners.GetOr(triangle,InvalidOffset);
        if (cluster >= records.size() || !index.Contains(owner.MeshletRoot,cluster)) throw std::invalid_argument("Meshlet patch triangle has no live render owner.");
        const auto &record = records[cluster];
        if (record.Topology || record.RefinedGroup != InvalidOffset || record.TriangleCount > 48u ||
            !std::ranges::contains(triangle_ids.subspan(record.TriangleOffset,record.TriangleCount),triangle))
            throw std::invalid_argument("Meshlet patch triangle disagrees with its render owner.");
        return cluster;
    };
    MeshletPatch patch;
    if (input.Changed.Storage.Slot != InvalidSlot) ForEachWorkElement(storage,input.Changed,[&](uint32_t triangle) {
        const auto cluster = cluster_of(triangle);
        const auto &record = records[cluster];
        const auto corners = buffers.MeshletVertexCorners.Get({record.VertexOffset,record.VertexCount});
        if (!input.StableKeys || contains(input.Removed,triangle) ||
            std::ranges::any_of(corners,[&](uint32_t corner) { return contains(input.ReplacedCorners,corner); }))
            patch.Clusters.push_back(cluster);
    });
    std::ranges::sort(patch.Clusters);
    patch.Clusters.erase(std::unique(patch.Clusters.begin(),patch.Clusters.end()),patch.Clusters.end());
    // Grouped partitions precede ungrouped ones, each in ascending order.
    std::map<std::pair<bool,uint32_t>,MeshletPatchPartition> partitions;
    for (const auto cluster : patch.Clusters) {
        const auto &record = records[cluster];
        if (!index.Contains(owner.PrimitiveRoot,record.Primitive) ||
            (record.GroupIndex != InvalidOffset && !index.Contains(owner.GroupRoot,record.GroupIndex)))
            throw std::invalid_argument("Meshlet patch cluster names a foreign primitive or group.");
        const bool grouped = record.GroupIndex != InvalidOffset;
        partitions[{!grouped,grouped ? record.GroupIndex : record.Primitive}] = {.Group=record.GroupIndex,.Primitive=record.Primitive};
    }
    const auto &triangle_blocks = meshes.Arenas().Triangles.Blocks;
    const auto triangle_owner = meshes.Get(owner.StoreId).TriangleData.Index;
    // Every partition is spatially local to one existing simplification group.
    // It takes a deterministic source leaf, and span repair redistributes overflow.
    const auto place = [&](uint32_t triangle, uint32_t source) {
        const auto &block = triangle_blocks.Get({triangle/MeshElementBlockSize,1u})[0];
        if (block.Owner != triangle_owner || !(block.Live[(triangle%MeshElementBlockSize)/32u] & (1u << (triangle%32u)))) {
            throw std::invalid_argument("Meshlet patch triangle is not live in its mesh.");
        }
        const auto cluster = cluster_of(source);
        if (!std::ranges::binary_search(patch.Clusters,cluster)) throw std::invalid_argument("Meshlet patch triangle keeps its source cluster.");
        const auto &record = records[cluster];
        const bool grouped = record.GroupIndex != InvalidOffset;
        auto &partition = partitions.at({!grouped,grouped ? record.GroupIndex : record.Primitive});
        const auto leaf = buffers.MeshletLodLeaves.Get({cluster,1u})[0];
        if (!index.Contains(owner.NodeRoot,leaf) || buffers.LodNodes.Get({leaf,1u})[0].ChildCount ||
            !index.Contains(buffers.LodNodes.Get({leaf,1u})[0].MeshletRoot,cluster))
            throw std::invalid_argument("Meshlet patch cluster disagrees with its traversal leaf.");
        partition.Leaf = std::min(partition.Leaf,leaf);
        partition.Elements.push_back(triangle);
    };
    for (const auto cluster : patch.Clusters) {
        const auto &record = records[cluster];
        for (const auto triangle : triangle_ids.subspan(record.TriangleOffset,record.TriangleCount))
            if (!contains(input.Removed,triangle)) place(triangle,triangle);
    }
    for (uint32_t i = 0u; i < input.Added.Count; ++i) place(input.Added.Offset+i,input.Sources[i]);
    for (auto &[key,partition] : partitions) {
        if (partition.Group != InvalidOffset) patch.Groups.push_back(partition.Group);
        if (!partition.Elements.empty()) patch.Partitions.push_back(std::move(partition));
    }
    return patch;
}

std::vector<uint32_t> AdoptMeshletFragments(state::Scene &r, mtl::ComputeChain &chain, MeshBuffers &owner, std::span<MeshBuffers> fragments,
                                            std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks) {
    const profile::CpuScope scope{"MeshletAdoptFragments"};
    const auto topology = owner.RenderTopology;
    if (topology>2u || fragments.size()!=jobs.size() || owner.PrimitiveRoot==InvalidOffset || owner.NodeRoot==InvalidOffset) {
        throw std::invalid_argument("Meshlet fragments require one matching canonical owner and job each.");
    }
    auto &buffers = r.Context.get<GpuBuffers>();
    std::vector<uint32_t> added;
    std::vector<Range> ranges;
    std::map<uint32_t,uint64_t> element_counts;
    for (uint32_t p=0u; p<fragments.size(); ++p) {
        const auto &fragment = fragments[p];
        const auto &job = jobs[p];
        if (&fragment == &owner || fragment.StoreId != owner.StoreId || fragment.PrimitiveRoot != InvalidOffset ||
            fragment.PrimitiveRoutes.Count || fragment.NodeRoot != InvalidOffset || fragment.GroupRoot != InvalidOffset ||
            fragment.Primitives.Count || fragment.LodNodes.Count || fragment.ClusterGroups.Count || fragment.MeshRecord.Count ||
            fragment.RenderTopology != InvalidOffset || fragment.ElementMeshletBlockCount ||
            fragment.Level0Count != fragment.Meshlets.Count || fragment.MeshletRoot != InvalidOffset ||
            job.First!=fragment.Meshlets.Offset || job.Count!=fragment.Meshlets.Count ||
            !buffers.ActiveMeshlets.Contains(owner.PrimitiveRoot,job.Primitive) ||
            (job.Group!=InvalidOffset && !buffers.ActiveMeshlets.Contains(owner.GroupRoot,job.Group)) ||
            (job.Count && (!buffers.ActiveMeshlets.Contains(owner.NodeRoot,job.Leaf) || buffers.LodNodes.Get({job.Leaf,1u})[0].ChildCount)))
            throw std::invalid_argument("Meshlet adoption requires independently owned, unpublished fragments.");
        for (uint32_t i=0u; i<job.Count; ++i) {
            if (buffers.ActiveMeshlets.Contains(owner.MeshletRoot,job.First+i)) throw std::invalid_argument("Meshlet adoption repeats an owned cluster.");
            added.push_back(job.First+i);
        }
        ranges.push_back(fragment.Meshlets);
        if (topology!=0u) element_counts[job.Primitive]+=fragment.MeshletTriangles.Count;
    }
    if (added.size() > UINT32_MAX-owner.Level0Count) throw std::length_error("Meshlet adoption exceeds canonical cluster capacity.");
    for (const auto &[primitive,count]:element_counts)
        if (count>UINT32_MAX-buffers.Primitives.Get({primitive,1u})[0].TriangleCount) throw std::length_error("Element adoption exceeds primitive element capacity.");
    std::ranges::sort(added);
    if (added.empty()) return added;
    MeshletIndexEdit ownership{.Root=owner.MeshletRoot,.Added=added};
    buffers.ActiveMeshlets.Update(std::span{&ownership,1u});
    owner.MeshletRoot = ownership.Root;
    owner.Level0Count += uint32_t(added.size());
    ++owner.MeshletRevision;
    for (const auto &[primitive,count]:element_counts) buffers.Primitives.GetMutable({primitive,1u})[0].TriangleCount+=uint32_t(count);
    for (const auto &job : jobs) std::ranges::fill(buffers.MeshletLodLeaves.GetMutable({job.First,job.Count}),job.Leaf);
    // The owner now owns every new payload, so exception cleanup finds no provisional range to release.
    for (auto &fragment : fragments) fragment = {};
    PublishMeshletOwners(r,chain,owner,ranges,blocks);
    std::vector<uint32_t> meshlet_blocks;
    for (const auto id : added) if (meshlet_blocks.empty() || meshlet_blocks.back()!=id/256u) meshlet_blocks.push_back(id/256u);
    buffers.PosedMeshletBounds.UpdateBlocks(owner.StoreId,owner.MeshletRevision,meshlet_blocks,
        [&](uint32_t block) { return buffers.ActiveMeshlets.HasBlock(owner.MeshletRoot,block); },owner.RenderTopology);
    return added;
}
