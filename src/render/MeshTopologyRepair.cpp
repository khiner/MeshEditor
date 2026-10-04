#include "Profile.h"
#include "SortUnique.h"
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

// One owner's partitions and the finest clusters they retire.
struct OwnerFragments {
    MeshStore::Record *Owner;
    std::span<const MeshletPatchPartition> Partitions;
    std::span<const uint32_t> Retired{};
};

// Every topology seeds, builds and adopts the same primitive-bound fragments, and every owner's fragments share one build.
// Reserve the seeds and builder work together before recording any dispatch.
std::vector<FragmentBuild> BuildFragments(state::Scene &r,mtl::ComputeChain &chain,std::span<const OwnerFragments> owners,
                                          std::span<MeshletBuildSource> fresh={}) {
    auto &meshes=r.Context.get<MeshStore>();
    size_t fragment_count=0u;
    for (const auto &owner:owners) fragment_count+=owner.Partitions.size();
    std::vector<MeshStore::Record> fragments(fragment_count);
    try {
        std::vector<MeshletBuildSource> sources;
        sources.reserve(fragment_count+fresh.size());
        std::vector<std::vector<uint32_t>> blocks(owners.size());
        // Each fragment's partition, element domain and seed block count.
        struct Seed { const MeshletPatchPartition *Partition; uint32_t Domain, Blocks; };
        std::vector<Seed> seeds;
        seeds.reserve(fragment_count);
        uint64_t seed_words=0u;
        for (uint32_t o=0u;o<owners.size();++o) {
            auto &owner=*owners[o].Owner;
            const auto topology=owner.RenderTopology;
            const auto domain=meshes.WithRenderDomain(owner,topology,[](const auto &arena, ElementSetRef) { return arena.Capacity(); });
            auto &owner_blocks=blocks[o];
            for (const auto &partition:owners[o].Partitions) {
                const auto first=owner_blocks.size();
                for (const auto element:partition.Elements) PushBlock(owner_blocks,element,first);
                const auto &seed=seeds.emplace_back(Seed{&partition,domain,topology==0u ? UniqueTail(owner_blocks,first) : uint32_t(owner_blocks.size()-first)});
                seed_words+=ElementWorkWords(domain,seed.Blocks);
                auto &fragment=fragments[sources.size()];
                fragment.StoreId=owner.StoreId;
                sources.push_back({.Destination=&fragment,.Topology=topology,
                    .ElementCount=uint32_t(partition.Elements.size()),.Owner=&owner,.Primitive=partition.Primitive,.Group=partition.Group});
            }
            if (!std::ranges::is_sorted(owner_blocks)) std::ranges::sort(owner_blocks);
            owner_blocks.erase(std::unique(owner_blocks.begin(),owner_blocks.end()),owner_blocks.end());
        }
        sources.insert(sources.end(),fresh.begin(),fresh.end());
        chain.Scratch.ReserveAdditional(seed_words+MeshletBuildScratchWords(meshes,sources));
        for (uint32_t i=0u;i<fragment_count;++i)
            sources[i].Elements=SeedElementWorkHandles(chain.Scratch,seeds[i].Domain,seeds[i].Partition->Elements,seeds[i].Blocks);
        BuildGpuMeshlets(r,chain,sources);
        std::vector<FragmentBuild> results(owners.size());
        for (uint32_t o=0u,first=0u;o<owners.size();++o) {
            auto &owner=*owners[o].Owner;
            const auto owner_partitions=owners[o].Partitions;
            RetireMeshletOwners(r,owner,owners[o].Retired);
            std::vector<MeshletPatchAdoptJob> jobs;
            jobs.reserve(owner_partitions.size());
            auto &result=results[o];
            result.Runs.reserve(owner_partitions.size());
            for (uint32_t i=0u;i<owner_partitions.size();++i) {
                const auto &partition=owner_partitions[i];
                const auto range=fragments[first+i].Meshlets;
                jobs.push_back({.First=range.Offset,.Count=range.Count,
                    .Group=partition.Group,.Primitive=partition.Primitive,.Leaf=partition.Leaf});
                result.Runs.push_back({range.Offset,range.Count,partition.Primitive,true});
            }
            result.Added=AdoptMeshletFragments(r,chain,owner,std::span{fragments}.subspan(first,owner_partitions.size()),jobs,blocks[o]);
            first+=uint32_t(owner_partitions.size());
        }
        return results;
    } catch (...) {
        std::vector<MeshStore::Record *> released;
        for (auto &fragment:fragments) released.push_back(&fragment);
        meshes.ReleaseRender(released);
        throw;
    }
}

// Removes the sorted elements from the point or line clusters that draw them, on the host, and returns the refit of the owner's nodes above them.
LodNodeRefit RepairElementMeshletDeletion(state::Scene &r,mtl::ComputeChain &chain,MeshStore::Record &owner,std::span<const uint32_t> elements) {
    const profile::CpuScope scope{"ElementMeshletDeletion"};
    auto &buffers=r.Context.get<GpuBuffers>();
    auto &meshes=r.Context.get<MeshStore>();
    auto &render=meshes.Render();
    const auto topology=owner.RenderTopology, endpoints=topology==1u ? 2u : 1u, origin=owner.ElementMeshletOrigin;
    if ((topology!=1u && topology!=2u) || !owner.ElementMeshletBlockCount) throw std::invalid_argument("Element deletion requires canonical point or line meshlet ownership.");
    const auto owners=render.ElementMeshlets[topology].View();
    std::vector<uint32_t> clusters;
    for (const auto element:elements) {
        const auto cluster=owners.GetOr(element,InvalidOffset);
        if (cluster>=render.Meshlets.Buffer.Count<MeshletRecord>() || !render.ActiveMeshlets.Contains(owner.MeshletRoot,cluster)) {
            throw std::invalid_argument("Deleted elements have no owned meshlets.");
        }
        clusters.push_back(cluster);
    }
    std::ranges::sort(clusters);
    clusters.erase(std::unique(clusters.begin(),clusters.end()),clusters.end());
    for (const auto cluster:clusters) {
        const auto &record=render.Meshlets.Get({cluster,1u})[0];
        if (record.Topology!=topology || record.RefinedGroup!=InvalidOffset || !record.TriangleCount || record.TriangleCount>16u ||
            record.VertexCount!=record.TriangleCount*endpoints || !render.ActiveMeshlets.Contains(owner.PrimitiveRoot,record.Primitive))
            throw std::invalid_argument("Element deletion touches a foreign meshlet or primitive.");
    }
    RetireMeshletOwners(r,owner,clusters);
    std::vector<uint32_t> kept,empty,kept_blocks;
    for (const auto cluster:clusters) {
        auto record=render.Meshlets.Get({cluster,1u})[0];
        const auto ids=render.MeshletTriangleIds.GetMutable({record.TriangleOffset,record.TriangleCount});
        const auto references=render.MeshletVertexCorners.GetMutable({record.VertexOffset,record.VertexCount});
        uint32_t count=0u;
        for (uint32_t n=0u;n<record.TriangleCount;++n) {
            const auto id=ids[n];
            if (std::ranges::binary_search(elements,origin+id)) continue;
            ids[count]=id;
            for (uint32_t c=0u;c<endpoints;++c) references[count*endpoints+c]=references[n*endpoints+c];
            kept_blocks.push_back((origin+id)/MeshElementBlockSize);
            ++count;
        }
        render.Primitives.GetMutable({record.Primitive,1u})[0].TriangleCount-=record.TriangleCount-count;
        // Storage retirement releases an emptied cluster's whole payload.
        if (!count) {
            empty.push_back(cluster);
            continue;
        }
        render.MeshletTriangleIds.Release({record.TriangleOffset+count,record.TriangleCount-count});
        render.MeshletVertexCorners.Release({record.VertexOffset+count*endpoints,(record.TriangleCount-count)*endpoints});
        record.TriangleCount=count;
        record.VertexCount=count*endpoints;
        render.Meshlets.GetMutable({cluster,1u})[0]=record;
        kept.push_back(cluster);
    }
    std::ranges::sort(kept_blocks);
    kept_blocks.erase(std::unique(kept_blocks.begin(),kept_blocks.end()),kept_blocks.end());
    PublishMeshletOwners(r,chain,owner,HandleRuns(kept),kept_blocks);
    if (!kept.empty()) {
        const MeshletBoundsRefitJob job{&owner,&chain.Scratch,SeedElementWorkHandles(chain.Scratch,render.Meshlets.Buffer.Count<MeshletRecord>(),kept)};
        RefitCanonicalMeshletBounds(r,chain,std::span{&job,1u});
    }
    auto refit=EditLodNodes(r,chain,owner,empty,{},kept);
    RetireMeshletStorage(r,owner,empty);
    if (empty.empty()) ++owner.MeshletRevision;
    UpdatePosedMeshletBlocks(r,owner,clusters);
    buffers.PreludeStale=true;
    return refit;
}
} // namespace

void RepairElementMeshlets(state::Scene &r, mtl::ComputeChain &chain, std::span<const ElementMeshletRepair> repairs) {
    auto &buffers=r.Context.get<GpuBuffers>();
    auto &meshes=r.Context.get<MeshStore>();
    auto &render=meshes.Render();
    const auto &a=meshes.Arenas();
    std::vector<std::vector<MeshletPatchPartition>> partitions(repairs.size());
    std::vector<OwnerFragments> additions;
    std::vector<LodNodeRefit> refits;
    for (uint32_t i=0u;i<repairs.size();++i) {
        auto &owner=meshes.WriteRecord(repairs[i].StoreId);
        const auto store_id=owner.StoreId, topology=owner.RenderTopology;
        const bool lines=topology==1u;
        const auto &record=meshes.Get(store_id);
        const auto owners=render.ElementMeshlets[topology].View();
        const auto blocks=(lines ? a.EdgeHalfedges.Blocks : a.Vertices.Blocks).Buffer.GetSpan<MeshElementBlock>();
        const auto set=lines ? record.EdgeData.Index : record.Vertices.Index;
        const auto capacity=lines ? a.EdgeHalfedges.Capacity() : a.Vertices.Capacity();
        const auto records=render.Meshlets.Buffer.GetSpan<MeshletRecord>();
        // Every live element draws through exactly one cluster.
        std::vector<uint32_t> added,removed;
        for (const auto element:repairs[i].Elements) {
            if (element>=capacity || element/MeshElementBlockSize>=blocks.size()) throw std::invalid_argument("Element repair names a handle outside its domain.");
            const auto &block=blocks[element/MeshElementBlockSize];
            const bool live=block.Owner==set && (block.Live[(element%MeshElementBlockSize)/32u] & (1u<<(element%32u)));
            const auto cluster=owners.GetOr(element,InvalidOffset);
            if (cluster!=InvalidOffset && (cluster>=records.size() || !render.ActiveMeshlets.Contains(owner.MeshletRoot,cluster) ||
                records[cluster].Topology!=topology))
                throw std::invalid_argument("Element owner names a foreign meshlet.");
            if (live && cluster==InvalidOffset) added.push_back(element);
            else if (!live && cluster!=InvalidOffset) removed.push_back(element);
        }
        for (auto *list:{&added,&removed}) SortUnique(*list);
        if (!removed.empty()) refits.push_back(RepairElementMeshletDeletion(r,chain,owner,removed));
        if (added.empty()) continue;
        // A line draws with the primitive of its first endpoint, where its first halfedge starts.
        const auto material_of=[&](uint32_t element) {
            const auto vertex=lines ? a.FaceCorners.Get({a.OppositeHalfedges.Get({a.EdgeHalfedges.Get({element,1u})[0],1u})[0],1u})[0] : element;
            return record.VertexPrimitivesReady ? a.VertexPrimitives.Get(vertex) : 0u;
        };
        std::map<uint32_t,std::vector<uint32_t>> by_material;
        for (const auto element:added) by_material[material_of(element)].push_back(element);
        auto &owner_partitions=partitions[i];
        owner_partitions.reserve(by_material.size());
        for (auto &[material,members]:by_material) {
            const auto primitive=EnsureMeshletPrimitive(r,owner,material);
            owner_partitions.push_back({.Primitive=primitive,.Leaf=render.Primitives.Get({primitive,1u})[0].LodFinestNode,.Elements=std::move(members)});
        }
        additions.push_back({&owner,owner_partitions});
    }
    if (!additions.empty()) {
        const auto built=BuildFragments(r,chain,additions);
        for (uint32_t o=0u;o<additions.size();++o) refits.push_back(EditLodNodes(r,chain,*additions[o].Owner,{},built[o].Runs,{}));
        buffers.PreludeStale=true;
    }
    RecordLodNodeRefits(r,chain,refits);
}

namespace {
// One owner's triangle edit.
struct TriangleRepair {
    state::Entity Entity;
    MeshletPatchInput Input;
};

// Repairs every owner's triangle render with one fragment build and one submit, before which the host cannot read the fragments' records.
void RepairTriangleRender(state::Scene &r,mtl::ComputeChain &chain,std::span<const TriangleRepair> repairs,std::span<MeshletBuildSource> fresh) {
    const profile::CpuScope scope{"TriangleRenderRepair"};
    auto &meshes=r.Context.get<MeshStore>();
    std::vector<state::Entity> entities;
    std::vector<MeshletPatch> patches;
    for (const auto &repair:repairs) {
        auto patch=PlanMeshletPatch(r,RecordOf(r,repair.Entity),chain.Scratch,repair.Input);
        if (patch.Clusters.empty() && patch.Partitions.empty()) continue;
        profile::RecordCounter("EditTouchedMeshlets", patch.Clusters.size());
        profile::RecordCounter("EditPatchPartitions", patch.Partitions.size());
        entities.push_back(repair.Entity);
        patches.push_back(std::move(patch));
    }
    if (patches.empty()) return BuildGpuMeshlets(r,chain,fresh);
    std::vector<OwnerFragments> owners;
    owners.reserve(patches.size());
    for (uint32_t i=0u;i<patches.size();++i) owners.push_back({&EditRecordOf(r,entities[i]),patches[i].Partitions,patches[i].Clusters});
    const auto built=BuildFragments(r,chain,owners,fresh);
    std::vector<ClusterGroupSeeds> seeds;
    seeds.reserve(patches.size());
    for (uint32_t i=0u;i<patches.size();++i) seeds.push_back({entities[i],patches[i].Groups});
    auto stale=InvalidateClusterGroups(r,seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i=0u;i<patches.size();++i) {
        const auto &patch=patches[i];
        profile::RecordCounter("TopologyAddedMeshlets", built[i].Added.size());
        r.Context.get<GpuSceneState>().LodDemand.insert(entities[i]);
        // Retired members leave their groups, and the remaining stale members refit.
        std::erase_if(stale[i],[&](uint32_t cluster) { return std::ranges::binary_search(patch.Clusters,cluster); });
        refits.push_back(EditLodNodes(r,chain,*owners[i].Owner,patch.Clusters,built[i].Runs,stale[i]));
    }
    RecordLodNodeRefits(r,chain,refits);
    // The host reads the fragments' records once the build has emitted them.
    chain.Submit();
    for (uint32_t i=0u;i<patches.size();++i) {
        auto &owner=*owners[i].Owner;
        ReplaceGroupClusters(meshes.Render(),patches[i].Clusters,built[i].Added);
        ReplaceMeshletSpatial(r,owner,patches[i].Clusters,built[i].Added);
        RetireMeshletStorage(r,owner,patches[i].Clusters);
    }
}
} // namespace

void RepairTopologyRender(state::Scene &r,mtl::ComputeChain &chain,std::span<const std::pair<state::Entity,const MeshTopologyEdit *>> edits,
                          std::span<MeshletBuildSource> fresh) {
    const auto &meshes=r.Context.get<const MeshStore>();
    std::vector<TriangleRepair> repairs;
    repairs.reserve(edits.size());
    for (const auto &[entity,topology]:edits) {
        const auto &record=meshes.Get(topology->StoreId);
        // Unchanged uniform keys can retain their meshlets unless their vertex
        // representative was a corner retired by this topology edit.
        const bool stable_keys = (topology->Op == MeshTopologyOp::InsetRegion || topology->Op == MeshTopologyOp::InsetIndividual) &&
            !(record.CornerAttributes & MeshAttributeBit_Normal) &&
            topology->OriginalClassMode == record.Classification &&
            record.Classification != uint32_t(CornerClassMode::Mixed);
        // Patch planning reads the sources before the repair allocates chain scratch.
        repairs.push_back({entity,{
            .Changed=topology->ChangedTriangles.Triangles,.Removed=topology->Output->Replaced[1],.ReplacedCorners=topology->Output->Replaced[0],
            .Added={topology->FirstTriangle,topology->AddedTriangleCount},.Sources=chain.Scratch.Get(topology->SourceTriangles),.StableKeys=stable_keys,
        }});
    }
    RepairTriangleRender(r,chain,repairs,fresh);
}

void RepairShadingRender(state::Scene &r,mtl::ComputeChain &chain,std::span<const std::pair<state::Entity,FaceTriangles>> changes) {
    auto &buffers=r.Context.get<GpuBuffers>();
    std::vector<TriangleRepair> repairs;
    std::vector<state::Entity> entities;
    for (const auto &[entity,changed]:changes) {
        if (!changed.Count) continue;
        const auto &owner=RecordOf(r,entity);
        if (owner.ExtrasFaces.Count || owner.PrimitiveRoot==InvalidOffset) throw std::invalid_argument("Local shading repair requires published render ownership.");
        // The descriptor's canonical values reflect the changed corner class.
        RefreshMeshBinding(r,owner.StoreId);
        repairs.push_back({entity,{.Changed=changed.Triangles}});
        entities.push_back(entity);
    }
    if (repairs.empty()) return;
    RepairTriangleRender(r,chain,repairs,{});
    buffers.PreludeStale=true;
    RepointMeshInstances(r,entities);
}
