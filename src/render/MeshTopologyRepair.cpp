#include "Profile.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletPatchWork.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletOwners.h"
#include "render/MeshletStorage.h"
#include "render/MeshletSpatial.h"
#include "render/ClusterLodRepair.h"
#include "render/LodNodeEdit.h"
#include "render/GpuBuffers.h"
#include "render/GpuBufferOps.h"
#include "render/GpuSceneState.h"
#include "render/ElementWorkOps.h"
#include "render/SceneUpdates.h"
#include "render/MeshletBoundsRefit.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopologyEdit.h"
#include "gpu/CornerClassMode.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"
#include <map>
#include <vector>

namespace {
// Appends the handle's block unless the values from `first` on already end with it.
void PushBlock(std::vector<uint32_t> &blocks, uint32_t handle, size_t first) {
    const auto block=handle/MeshElementBlockSize;
    if (blocks.size()==first || blocks.back()!=block) blocks.push_back(block);
}
// Sorts and dedupes the values from `first` on, and returns how many remain.
uint32_t UniqueTail(std::vector<uint32_t> &values, size_t first) {
    std::ranges::sort(values.begin()+first,values.end());
    values.erase(std::unique(values.begin()+first,values.end()),values.end());
    return uint32_t(values.size()-first);
}
// Ascending runs of consecutive handles.
std::vector<Range> HandleRuns(std::span<const uint32_t> sorted) {
    std::vector<Range> runs;
    for (const auto handle : sorted) {
        if (!runs.empty() && runs.back().Offset+runs.back().Count == handle) ++runs.back().Count;
        else runs.push_back({handle,1u});
    }
    return runs;
}

struct FragmentBuild {
    std::vector<uint32_t> Added;
    std::vector<LodClusterRun> Runs;
};

// Every topology seeds, builds and adopts the same primitive-bound fragments.
// Reserve the seeds and builder work together before recording any dispatch.
FragmentBuild BuildFragments(state::Scene &r,mtl::ComputeChain &chain,MeshBuffers &owner,
                             std::span<const MeshletPatchPartition> partitions,std::span<const uint32_t> retired={},
                             std::span<MeshletBuildSource> fresh={}) {
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &a=meshes.Arenas();
    const auto topology=owner.RenderTopology;
    const auto domain=topology==0u ? a.Triangles.Capacity() : topology==1u ? a.EdgeHalfedges.Capacity() : a.Vertices.Capacity();
    const auto mesh=BuildMeshRecord(buffers,owner,meshes,owner.StoreId,topology==0u,topology==1u);
    std::vector<MeshBuffers> fragments(partitions.size());
    try {
        std::vector<MeshletBuildSource> sources;
        sources.reserve(partitions.size()+fresh.size());
        std::vector<uint32_t> blocks,seed_blocks;
        seed_blocks.reserve(partitions.size());
        uint64_t seed_words=0u;
        for (uint32_t i=0u;i<partitions.size();++i) {
            const auto &partition=partitions[i];
            const auto first=blocks.size();
            for (const auto element:partition.Elements) PushBlock(blocks,element,first);
            seed_blocks.push_back(topology==0u ? UniqueTail(blocks,first) : uint32_t(blocks.size()-first));
            fragments[i].Vertices=owner.Vertices;
            sources.push_back({.Destination=&fragments[i],.Mesh=mesh,.StoreId=owner.StoreId,.Topology=topology,
                .ElementCount=uint32_t(partition.Elements.size()),.Owner=&owner,.Primitive=partition.Primitive,.Group=partition.Group});
            seed_words+=ElementWorkWords(domain,seed_blocks.back());
        }
        sources.insert(sources.end(),fresh.begin(),fresh.end());
        chain.Scratch.ReserveAdditional(seed_words+MeshletBuildScratchWords(meshes,sources));
        for (uint32_t i=0u;i<partitions.size();++i)
            sources[i].Elements=SeedElementWorkHandles(chain.Scratch,domain,partitions[i].Elements,seed_blocks[i]);
        if (!std::ranges::is_sorted(blocks)) std::ranges::sort(blocks);
        blocks.erase(std::unique(blocks.begin(),blocks.end()),blocks.end());
        BuildGpuMeshlets(r,chain,sources);
        RetireMeshletOwners(r,owner,retired);
        std::vector<MeshletPatchAdoptJob> jobs;
        jobs.reserve(partitions.size());
        FragmentBuild result;
        result.Runs.reserve(partitions.size());
        for (uint32_t i=0u;i<partitions.size();++i) {
            const auto &partition=partitions[i];
            const auto range=fragments[i].Meshlets;
            jobs.push_back({.First=range.Offset,.Count=range.Count,
                .Group=partition.Group,.Primitive=partition.Primitive,.Leaf=partition.Leaf});
            result.Runs.push_back({range.Offset,range.Count,partition.Primitive,true});
        }
        result.Added=AdoptMeshletFragments(r,chain,owner,fragments,jobs,blocks);
        return result;
    } catch (...) {
        for (auto &fragment:fragments) buffers.ReleaseMeshlets(fragment);
        throw;
    }
}

// Removes the sorted elements from the point or line clusters that draw them, on the host.
void RepairElementMeshletDeletion(state::Scene &r,mtl::ComputeChain &chain,MeshBuffers &owner,std::span<const uint32_t> elements) {
    const profile::CpuScope scope{"ElementMeshletDeletion"};
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto topology=owner.RenderTopology, endpoints=topology==1u ? 2u : 1u, origin=owner.ElementMeshletOrigin;
    if ((topology!=1u && topology!=2u) || !owner.ElementMeshletBlockCount) throw std::invalid_argument("Element deletion requires canonical point or line meshlet ownership.");
    const auto owners=buffers.ElementMeshlets[topology].View();
    std::vector<uint32_t> clusters;
    for (const auto element:elements) {
        const auto cluster=owners.GetOr(element,InvalidOffset);
        if (cluster>=buffers.Meshlets.Buffer.Count<MeshletRecord>() || !buffers.ActiveMeshlets.Contains(owner.MeshletRoot,cluster)) {
            throw std::invalid_argument("Deleted elements have no owned meshlets.");
        }
        clusters.push_back(cluster);
    }
    std::ranges::sort(clusters);
    clusters.erase(std::unique(clusters.begin(),clusters.end()),clusters.end());
    for (const auto cluster:clusters) {
        const auto &record=buffers.Meshlets.Get({cluster,1u})[0];
        if (record.Topology!=topology || record.RefinedGroup!=InvalidOffset || !record.TriangleCount || record.TriangleCount>16u ||
            record.VertexCount!=record.TriangleCount*endpoints || !buffers.ActiveMeshlets.Contains(owner.PrimitiveRoot,record.Primitive))
            throw std::invalid_argument("Element deletion touches a foreign meshlet or primitive.");
    }
    RetireMeshletOwners(r,owner,clusters);
    std::vector<uint32_t> kept,empty,kept_blocks;
    for (const auto cluster:clusters) {
        auto record=buffers.Meshlets.Get({cluster,1u})[0];
        const auto ids=buffers.MeshletTriangleIds.GetMutable({record.TriangleOffset,record.TriangleCount});
        const auto references=buffers.MeshletVertexCorners.GetMutable({record.VertexOffset,record.VertexCount});
        uint32_t count=0u;
        for (uint32_t n=0u;n<record.TriangleCount;++n) {
            const auto id=ids[n];
            if (std::ranges::binary_search(elements,origin+id)) continue;
            ids[count]=id;
            for (uint32_t c=0u;c<endpoints;++c) references[count*endpoints+c]=references[n*endpoints+c];
            kept_blocks.push_back((origin+id)/MeshElementBlockSize);
            ++count;
        }
        buffers.Primitives.GetMutable({record.Primitive,1u})[0].TriangleCount-=record.TriangleCount-count;
        // Storage retirement releases an emptied cluster's whole payload.
        if (!count) {
            empty.push_back(cluster);
            continue;
        }
        buffers.MeshletTriangleIds.Release({record.TriangleOffset+count,record.TriangleCount-count});
        buffers.MeshletVertexCorners.Release({record.VertexOffset+count*endpoints,(record.TriangleCount-count)*endpoints});
        record.TriangleCount=count;
        record.VertexCount=count*endpoints;
        buffers.Meshlets.GetMutable({cluster,1u})[0]=record;
        kept.push_back(cluster);
    }
    std::ranges::sort(kept_blocks);
    kept_blocks.erase(std::unique(kept_blocks.begin(),kept_blocks.end()),kept_blocks.end());
    PublishMeshletOwners(r,chain,owner,HandleRuns(kept),kept_blocks);
    if (!kept.empty()) {
        const MeshletBoundsRefitJob job{&owner,&chain.Scratch,SeedElementWorkHandles(chain.Scratch,buffers.Meshlets.Buffer.Count<MeshletRecord>(),kept)};
        RefitCanonicalMeshletBounds(r,std::span{&job,1u});
    }
    EditLodNodes(r,chain,owner,empty,{},kept);
    RetireMeshletStorage(r,owner,empty);
    if (empty.empty()) ++owner.MeshletRevision;
    std::vector<uint32_t> blocks;
    for (const auto cluster:clusters) if (blocks.empty() || blocks.back()!=cluster/256u) blocks.push_back(cluster/256u);
    buffers.PosedMeshletBounds.UpdateBlocks(owner.StoreId,owner.MeshletRevision,blocks,
        [&](uint32_t block) { return buffers.ActiveMeshlets.HasBlock(owner.MeshletRoot,block); },owner.RenderTopology);
    buffers.PreludeStale=true;
}
} // namespace

void RepairElementMeshlets(state::Scene &r, mtl::ComputeChain &chain, MeshBuffers &owner, std::span<const uint32_t> elements) {
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &a=meshes.Arenas();
    const auto store_id=owner.StoreId, topology=owner.RenderTopology;
    const bool lines=topology==1u;
    const auto &record=meshes.Get(store_id);
    const auto owners=buffers.ElementMeshlets[topology].View();
    const auto blocks=(lines ? a.EdgeHalfedges.Blocks : a.Vertices.Blocks).Buffer.GetSpan<MeshElementBlock>();
    const auto set=lines ? record.EdgeData.Index : record.Vertices.Index;
    const auto capacity=lines ? a.EdgeHalfedges.Capacity() : a.Vertices.Capacity();
    const auto records=buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    // Every live element draws through exactly one cluster.
    std::vector<uint32_t> added,removed;
    for (const auto element:elements) {
        if (element>=capacity || element/MeshElementBlockSize>=blocks.size()) throw std::invalid_argument("Element repair names a handle outside its domain.");
        const auto &block=blocks[element/MeshElementBlockSize];
        const bool live=block.Owner==set && (block.Live[(element%MeshElementBlockSize)/32u] & (1u<<(element%32u)));
        const auto cluster=owners.GetOr(element,InvalidOffset);
        if (cluster!=InvalidOffset && (cluster>=records.size() || !buffers.ActiveMeshlets.Contains(owner.MeshletRoot,cluster) ||
            records[cluster].Topology!=topology))
            throw std::invalid_argument("Element owner names a foreign meshlet.");
        if (live && cluster==InvalidOffset) added.push_back(element);
        else if (!live && cluster!=InvalidOffset) removed.push_back(element);
    }
    for (auto *list:{&added,&removed}) {
        std::ranges::sort(*list);
        list->erase(std::unique(list->begin(),list->end()),list->end());
    }
    if (!removed.empty()) RepairElementMeshletDeletion(r,chain,owner,removed);
    if (added.empty()) return;
    // A line draws with the primitive of its first endpoint, where its first halfedge starts.
    const auto material_of=[&](uint32_t element) {
        const auto vertex=lines ? a.FaceCorners.Get({a.OppositeHalfedges.Get({a.EdgeHalfedges.Get({element,1u})[0],1u})[0],1u})[0] : element;
        return record.VertexPrimitivesReady ? a.VertexPrimitives.Get(vertex) : 0u;
    };
    std::map<uint32_t,std::vector<uint32_t>> by_material;
    for (const auto element:added) by_material[material_of(element)].push_back(element);
    std::vector<MeshletPatchPartition> partitions;
    partitions.reserve(by_material.size());
    for (auto &[material,members]:by_material) {
        const auto primitive=EnsureMeshletPrimitive(r,owner,material);
        partitions.push_back({.Primitive=primitive,.Leaf=buffers.Primitives.Get({primitive,1u})[0].LodFinestNode,.Elements=std::move(members)});
    }
    const auto built=BuildFragments(r,chain,owner,partitions);
    EditLodNodes(r,chain,owner,{},built.Runs,{});
    buffers.PreludeStale=true;
}

namespace {
void RepairTriangleRender(state::Scene &r,mtl::ComputeChain &chain,state::Entity entity,const MeshletPatchInput &input,
                          std::span<MeshletBuildSource> fresh) {
    const profile::CpuScope scope{"TriangleRenderRepair"};
    auto &buffers=r.Context.get<GpuBuffers>();
    auto &owner=MeshBuffersOf(r,entity);
    const auto patch=PlanMeshletPatch(r,owner,chain.Scratch,input);
    if (patch.Clusters.empty() && patch.Partitions.empty()) return BuildGpuMeshlets(r,chain,fresh);
    profile::RecordCounter("EditTouchedMeshlets", patch.Clusters.size());
    profile::RecordCounter("EditPatchPartitions", patch.Partitions.size());
    const auto built=BuildFragments(r,chain,owner,patch.Partitions,patch.Clusters,fresh);
    profile::RecordCounter("TopologyAddedMeshlets", built.Added.size());
    r.Context.get<GpuSceneState>().LodDemand.insert(entity);
    // Retired members leave their groups; the remaining stale members refit.
    auto stale=InvalidateClusterGroups(r,entity,patch.Groups);
    std::erase_if(stale,[&](uint32_t cluster) { return std::ranges::binary_search(patch.Clusters,cluster); });
    EditLodNodes(r,chain,owner,patch.Clusters,built.Runs,stale);
    // The host reads the fragments' records once the build has emitted them.
    chain.Submit();
    ReplaceGroupClusters(buffers,patch.Clusters,built.Added);
    ReplaceMeshletSpatial(r,owner,patch.Clusters,built.Added);
    RetireMeshletStorage(r,owner,patch.Clusters);
}
} // namespace

void RepairTopologyRender(state::Scene &r,state::Entity entity,const MeshTopologyEdit &topology,std::span<MeshletBuildSource> fresh) {
    if (!topology.Output) return;
    auto &owner=MeshBuffersOf(r,entity);
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &a=meshes.Arenas();
    const auto &record=meshes.Get(topology.StoreId);
    owner.Vertices={{a.Vertices.First(record.Vertices),a.Vertices.Count(record.Vertices)},a.Vertices.Buffer.Slot};
    // Unchanged uniform keys can retain their meshlets unless their vertex
    // representative was a corner retired by this topology edit.
    const bool stable_keys = (topology.Op == MeshTopologyOp::InsetRegion || topology.Op == MeshTopologyOp::InsetIndividual) &&
        !(record.CornerAttributes & MeshAttributeBit_Normal) &&
        topology.OriginalClassMode == record.Classification &&
        record.Classification != uint32_t(CornerClassMode::Mixed);
    RepairTriangleRender(r,topology.Chain,entity,{
        .Changed=topology.ChangedTriangles.Triangles,.Removed=topology.Output->Replaced[1],.ReplacedCorners=topology.Output->Replaced[0],
        .Added={topology.FirstTriangle,topology.AddedTriangleCount},.Sources=topology.SourceTriangles.GetSpan<uint32_t>(),.StableKeys=stable_keys,
    },fresh);
}

void RepairShadingRender(state::Scene &r,mtl::ComputeChain &chain,state::Entity entity,const FaceTriangles &changed) {
    if (!changed.Count) return;
    auto &buffers=r.Context.get<GpuBuffers>();
    auto &owner=MeshBuffersOf(r,entity);
    if (owner.StoreId==InvalidOffset || owner.PrimitiveRoot==InvalidOffset) throw std::invalid_argument("Local shading repair requires published render ownership.");
    // Runtime slots rebuild on history restore, and the descriptor's canonical values reflect the changed corner class now.
    buffers.MeshRecords.GetMutable(owner.MeshRecord)[0]=BuildMeshRecord(
        buffers,owner,r.Context.get<const MeshStore>(),owner.StoreId,true,false);
    RepairTriangleRender(r,chain,entity,{.Changed=changed.Triangles},{});
    buffers.PreludeStale=true;
    RepointMeshInstances(r,std::span{&entity,1u});
}
