#include "render/MeshTopologyRepair.h"
#include "Profile.h"
#include "SortUnique.h"
#include "mesh/MeshTopologyEdit.h"
#include "metal/Dispatch.h"
#include "render/ClusterLodRepair.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBufferOps.h"
#include "render/GpuSceneState.h"
#include "render/LodNodeEdit.h"
#include "render/MeshletBoundsRefit.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletOwners.h"
#include "render/MeshletPatchWork.h"
#include "render/MeshletSpatial.h"
#include "render/MeshletStorage.h"
#include "render/SceneUpdates.h"
#include "state/Registry.h"
#include <map>

namespace {
// Appends the handle's block unless the values from `first` on already end with it.
void PushBlock(std::vector<uint32_t> &blocks, uint32_t handle, size_t first) {
    const auto block = handle / MeshElementBlockSize;
    if (blocks.size() == first || blocks.back() != block) blocks.push_back(block);
}
uint32_t UniqueTail(std::vector<uint32_t> &values, size_t first) {
    std::ranges::sort(values.begin() + first, values.end());
    values.erase(std::ranges::unique(values.begin() + first, values.end()).begin(), values.end());
    return uint32_t(values.size() - first);
}
// Ascending runs of consecutive handles.
std::vector<Range> HandleRuns(std::span<const uint32_t> sorted) {
    std::vector<Range> runs;
    for (const auto handle : sorted) {
        if (!runs.empty() && runs.back().Offset + runs.back().Count == handle) ++runs.back().Count;
        else runs.push_back({handle, 1u});
    }
    return runs;
}

struct FragmentBuild {
    std::vector<uint32_t> Added;
    std::vector<LodClusterRun> Runs;
};

struct OwnerFragments {
    MeshStore::Record *Owner;
    std::span<const MeshletPatchPartition> Partitions;
    std::span<const uint32_t> Retired{};
    uint32_t Topology{};
};

// Every topology seeds, builds and adopts the same primitive-bound fragments, and every owner's fragments share one build.
// Reserve the seeds and builder work together before recording any dispatch.
std::vector<FragmentBuild> BuildFragments(state::Registry &r, mtl::ComputeChain &chain, std::span<const OwnerFragments> owners, std::span<MeshletBuildSource> fresh = {}) {
    auto &meshes = r.Context.get<MeshStore>();
    size_t fragment_count = 0u;
    for (const auto &owner : owners) fragment_count += owner.Partitions.size();
    std::vector<MeshStore::Record> fragments(fragment_count);
    try {
        std::vector<MeshletBuildSource> sources;
        sources.reserve(fragment_count + fresh.size());
        std::vector<std::vector<uint32_t>> blocks(owners.size());
        struct Seed {
            const MeshletPatchPartition *Partition;
            uint32_t Domain, Blocks;
        };
        std::vector<Seed> seeds;
        seeds.reserve(fragment_count);
        uint64_t seed_words = 0u;
        for (uint32_t o = 0u; o < owners.size(); ++o) {
            auto &owner = *owners[o].Owner;
            const auto topology = owners[o].Topology;
            const auto domain = meshes.WithRenderDomain(owner, topology, [](const auto &arena, ElementSetRef set) { return arena.HandleBound(set); });
            auto &owner_blocks = blocks[o];
            for (const auto &partition : owners[o].Partitions) {
                const auto first = owner_blocks.size();
                for (const auto element : partition.Elements) PushBlock(owner_blocks, element, first);
                const auto &seed = seeds.emplace_back(Seed{&partition, domain, topology == 0u ? UniqueTail(owner_blocks, first) : uint32_t(owner_blocks.size() - first)});
                seed_words += ElementWorkWords(domain, seed.Blocks);
                auto &fragment = fragments[sources.size()];
                fragment.StoreId = owner.StoreId;
                sources.push_back({.Destination = &fragment, .Topology = topology, .ElementCount = uint32_t(partition.Elements.size()), .Owner = &owner, .Primitive = partition.Primitive, .Group = partition.Group});
            }
            if (!std::ranges::is_sorted(owner_blocks)) SortUnique(owner_blocks);
        }
        sources.insert(sources.end(), fresh.begin(), fresh.end());
        chain.Scratch.ReserveAdditional(seed_words + MeshletBuildScratchWords(meshes, sources));
        for (uint32_t i = 0u; i < fragment_count; ++i) {
            sources[i].Elements = SeedElementWorkHandles(chain.Scratch, seeds[i].Domain, seeds[i].Partition->Elements, seeds[i].Blocks);
        }
        BuildGpuMeshlets(r, chain, sources);
        std::vector<FragmentBuild> results(owners.size());
        for (uint32_t o = 0u, first = 0u; o < owners.size(); ++o) {
            auto &owner = *owners[o].Owner;
            const auto owner_partitions = owners[o].Partitions;
            RetireMeshletOwners(r, owner, owners[o].Retired, {});
            std::vector<MeshletPatchAdoptJob> jobs;
            jobs.reserve(owner_partitions.size());
            auto &result = results[o];
            result.Runs.reserve(owner_partitions.size());
            for (uint32_t i = 0u; i < owner_partitions.size(); ++i) {
                const auto &partition = owner_partitions[i];
                const auto range = fragments[first + i].Meshlets;
                jobs.push_back({.First = range.Offset, .Count = range.Count, .ElementCount = uint32_t(partition.Elements.size()), .Group = partition.Group, .Primitive = partition.Primitive, .Leaf = partition.Leaf});
                result.Runs.push_back({range.Offset, range.Count, partition.Primitive, true});
            }
            result.Added = AdoptMeshletFragments(r, chain, owner, owners[o].Topology, std::span{fragments}.subspan(first, owner_partitions.size()), jobs, blocks[o]);
            first += uint32_t(owner_partitions.size());
        }
        return results;
    } catch (...) {
        std::vector<MeshStore::Record *> released;
        released.reserve(fragments.size());
        for (auto &fragment : fragments) released.push_back(&fragment);
        meshes.ReleaseRender(released);
        throw;
    }
}

LodNodeRefit RepairElementMeshletDeletion(state::Registry &r, mtl::ComputeChain &chain, MeshStore::Record &owner, uint32_t topology, std::span<const uint32_t> elements) {
    const profile::CpuScope scope{"ElementMeshletDeletion"};
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    const auto endpoints = topology == 1u ? 2u : 1u;
    if ((topology != 1u && topology != 2u) || !owner.ElementMeshletBlockCounts[topology]) throw std::invalid_argument("Element deletion requires canonical point or line meshlet ownership.");
    const auto entries = meshes.WithRenderDomain(owner, topology, [](const auto &arena, ElementSetRef set) { return arena.Entries(set); });
    const auto owners = render.ElementMeshlets[topology].View(owner.ElementMeshlets[topology]);
    std::vector<uint32_t> clusters;
    for (const auto element : elements) {
        const auto cluster = owners.GetOr(element, InvalidOffset);
        if (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{cluster})) throw std::invalid_argument("Deleted elements have no owned meshlets.");
        clusters.push_back(cluster);
    }
    SortUnique(clusters);
    const auto address = [&](uint32_t cluster) { return render.Meshlets.Address(owner.MeshletData, ElementHandle{cluster}).Value(); };
    for (const auto cluster : clusters) {
        const auto &record = render.Meshlets.Get({address(cluster), 1u})[0];
        if (record.Topology != topology || record.RefinedGroup != InvalidOffset || !record.TriangleCount || record.TriangleCount > 16u ||
            record.VertexCount != record.TriangleCount * endpoints || !render.Primitives.Contains(owner.PrimitiveData, ElementHandle{record.Primitive})) {
            throw std::invalid_argument("Element deletion touches a foreign meshlet or primitive.");
        }
    }
    std::vector<uint32_t> addresses;
    addresses.reserve(clusters.size());
    for (const auto cluster : clusters) addresses.push_back(address(cluster));
    std::ranges::sort(addresses);
    RetireMeshletOwners(r, owner, clusters, elements);
    std::vector<uint32_t> kept, kept_addresses, empty, kept_blocks;
    std::vector<Range> id_tails, corner_tails;
    for (const auto cluster : clusters) {
        auto record = render.Meshlets.Get({address(cluster), 1u})[0];
        const auto ids = render.MeshletTriangleIds.GetMutable({owner.MeshletTriangles.Values.Offset + record.TriangleOffset, record.TriangleCount});
        const auto references = render.MeshletVertexCorners.GetMutable({owner.MeshletVertices.Values.Offset + record.VertexOffset, record.VertexCount});
        uint32_t count = 0u;
        for (uint32_t n = 0u; n < record.TriangleCount; ++n) {
            // An id whose block left the record's table names an erased element.
            const auto id = ids[n], local = id / MeshElementBlockSize;
            if (local >= entries.size() || entries[local] == InvalidOffset || std::ranges::binary_search(elements, id)) continue;
            ids[count] = id;
            for (uint32_t c = 0u; c < endpoints; ++c) references[count * endpoints + c] = references[n * endpoints + c];
            kept_blocks.push_back(local);
            ++count;
        }
        render.Primitives.GetMutable({render.Primitives.Address(owner.PrimitiveData, ElementHandle{record.Primitive}).Value(), 1u})[0].TriangleCount -= record.TriangleCount - count;
        // Storage retirement releases an emptied cluster's whole payload.
        if (!count) {
            empty.push_back(cluster);
            continue;
        }
        id_tails.push_back({record.TriangleOffset + count, record.TriangleCount - count});
        corner_tails.push_back({record.VertexOffset + count * endpoints, (record.TriangleCount - count) * endpoints});
        record.TriangleCount = count;
        record.VertexCount = count * endpoints;
        render.Meshlets.GetMutable({address(cluster), 1u})[0] = record;
        kept.push_back(cluster);
        kept_addresses.push_back(address(cluster));
    }
    meshes.ReleasePayloadRuns(owner, RenderPayload::TriangleIds, std::move(id_tails));
    meshes.ReleasePayloadRuns(owner, RenderPayload::VertexCorners, std::move(corner_tails));
    SortUnique(kept_blocks);
    std::ranges::sort(kept_addresses);
    std::vector<MeshletOwnersJob> owner_jobs;
    PrepareMeshletOwners(r, owner, topology, HandleRuns(kept), kept_blocks, owner_jobs);
    RecordMeshletOwners(r, chain, owner_jobs);
    if (!kept.empty()) {
        const MeshletBoundsRefitJob job{&owner, &chain.Scratch, SeedElementWorkHandles(chain.Scratch, render.Meshlets.Capacity(), kept_addresses)};
        RefitCanonicalMeshletBounds(r, chain, std::span{&job, 1u});
    }
    auto refit = EditLodNodes(r, owner, empty, {}, kept);
    RetireMeshletStorage(r, owner, empty);
    if (empty.empty()) ++owner.MeshletRevision;
    UpdatePosedMeshletBlocks(r, owner, addresses);
    return refit;
}
} // namespace

void RepairElementMeshlets(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementMeshletRepair> repairs) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    const auto &a = meshes.Arenas();
    std::vector<std::vector<MeshletPatchPartition>> partitions(repairs.size());
    std::vector<OwnerFragments> additions;
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < repairs.size(); ++i) {
        auto &owner = meshes.WriteRecord(repairs[i].StoreId);
        const auto store_id = owner.StoreId, topology = repairs[i].Topology;
        if (topology != 1u && topology != 2u) throw std::invalid_argument("Element repair requires line or point topology.");
        const bool lines = topology == 1u;
        const auto &record = meshes.Get(store_id);
        const auto owners = render.ElementMeshlets[topology].View(record.ElementMeshlets[topology]);
        const auto domain = lines ? MeshStore::ElementDomain::Edge : MeshStore::ElementDomain::Vertex;
        const auto halfedge = [&](uint32_t h) { return meshes.HandleAddress(store_id, MeshStore::ElementDomain::Halfedge, ElementHandle{h}).Value(); };
        const auto records = render.Meshlets.Buffer.GetSpan<MeshletRecord>();
        // Only loose edges and isolated vertices draw through these owner maps.
        // A live element can stop drawing here when it gains face or edge incidence.
        std::vector<uint32_t> added, removed;
        for (const auto element : repairs[i].Elements) {
            const bool live = meshes.IsLiveElement(store_id, lines ? Element::Edge : Element::Vertex, ElementHandle{element});
            const auto cluster = owners.GetOr(element, InvalidOffset);
            if (cluster != InvalidOffset && (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{cluster}) || records[render.Meshlets.Address(owner.MeshletData, ElementHandle{cluster}).Value()].Topology != topology)) {
                throw std::invalid_argument("Element owner names a foreign meshlet.");
            }
            const auto address = live ? meshes.HandleAddress(store_id, domain, ElementHandle{element}).Value() : InvalidOffset;
            const bool draws = live && (lines ? a.HalfedgeFaces.Get({halfedge(a.EdgeHalfedges.Get({address, 1u})[0]), 1u})[0] == InvalidOffset : a.VertexCorners.Get({address, 1u})[0].y == 0u);
            if (draws && cluster == InvalidOffset) added.push_back(element);
            else if (!draws && cluster != InvalidOffset) removed.push_back(element);
        }
        for (auto *list : {&added, &removed}) SortUnique(*list);
        if (!removed.empty()) refits.push_back(RepairElementMeshletDeletion(r, chain, owner, topology, removed));
        if (added.empty()) continue;
        // A line draws with the primitive of its first endpoint, where its first halfedge starts.
        const auto primitives = a.VertexPrimitives.View(record.VertexPrimitives);
        const auto material_of = [&](uint32_t element) {
            const auto vertex = lines ? a.FaceCorners.Get({halfedge(a.OppositeHalfedges.Get({halfedge(a.EdgeHalfedges.Get({meshes.HandleAddress(store_id, domain, ElementHandle{element}).Value(), 1u})[0]), 1u})[0]), 1u})[0] : element;
            return primitives.GetOr(vertex);
        };
        std::map<uint32_t, std::vector<uint32_t>> by_material;
        for (const auto element : added) by_material[material_of(element)].push_back(element);
        auto &owner_partitions = partitions[i];
        owner_partitions.reserve(by_material.size());
        for (auto &[material, members] : by_material) {
            std::ranges::sort(members);
            const auto primitive = EnsureMeshletPrimitive(r, owner, material, topology);
            owner_partitions.push_back({.Primitive = primitive, .Leaf = render.Primitives.Get({render.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive}).Value(), 1u})[0].LodFinestNode, .Elements = std::move(members)});
        }
        additions.push_back({&owner, owner_partitions, {}, topology});
    }
    if (!additions.empty()) {
        const auto built = BuildFragments(r, chain, additions);
        for (uint32_t o = 0u; o < additions.size(); ++o) refits.push_back(EditLodNodes(r, *additions[o].Owner, {}, built[o].Runs, {}));
    }
    RecordLodNodeRefits(r, chain, refits);
}

namespace {
struct TriangleRepair {
    state::Entity Entity;
    MeshletPatchInput Input;
};

// Repairs every owner's triangle render with one fragment build and one submit, before which the host cannot read the fragments' records.
void RepairTriangleRender(state::Registry &r, mtl::ComputeChain &chain, std::span<const TriangleRepair> repairs, std::span<MeshletBuildSource> fresh) {
    const profile::CpuScope scope{"TriangleRenderRepair"};
    auto &meshes = r.Context.get<MeshStore>();
    std::vector<state::Entity> entities;
    std::vector<MeshletPatch> patches;
    for (const auto &repair : repairs) {
        auto patch = PlanMeshletPatch(r, EditRecordOf(r, repair.Entity), chain.Scratch, repair.Input);
        if (patch.Clusters.empty() && patch.Partitions.empty()) continue;
        profile::RecordCounter("EditTouchedMeshlets", patch.Clusters.size());
        profile::RecordCounter("EditPatchPartitions", patch.Partitions.size());
        entities.push_back(repair.Entity);
        patches.push_back(std::move(patch));
    }
    if (patches.empty()) return BuildGpuMeshlets(r, chain, fresh);
    std::vector<OwnerFragments> owners;
    owners.reserve(patches.size());
    for (uint32_t i = 0u; i < patches.size(); ++i) owners.push_back({&EditRecordOf(r, entities[i]), patches[i].Partitions, patches[i].Clusters});
    const auto built = BuildFragments(r, chain, owners, fresh);
    std::vector<ClusterGroupSeeds> seeds;
    seeds.reserve(patches.size());
    for (uint32_t i = 0u; i < patches.size(); ++i) seeds.push_back({entities[i], patches[i].Groups});
    auto stale = InvalidateClusterGroups(r, seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < patches.size(); ++i) {
        const auto &patch = patches[i];
        profile::RecordCounter("TopologyAddedMeshlets", built[i].Added.size());
        r.emplace_or_replace<ClusterLodDemand>(entities[i]);
        // Remove retired members from their groups and refit retained stale members.
        std::erase_if(stale[i], [&](uint32_t cluster) { return std::ranges::binary_search(patch.Clusters, cluster); });
        refits.push_back(EditLodNodes(r, *owners[i].Owner, patch.Clusters, built[i].Runs, stale[i]));
    }
    RecordLodNodeRefits(r, chain, refits);
    // The host reads the fragments' records once the build has emitted them.
    chain.Submit();
    for (uint32_t i = 0u; i < patches.size(); ++i) {
        auto &owner = *owners[i].Owner;
        ReplaceGroupClusters(meshes, owner, patches[i].Clusters, built[i].Added);
        ReplaceMeshletSpatial(r, owner, patches[i].Clusters, built[i].Added);
        RetireMeshletStorage(r, owner, patches[i].Clusters);
    }
}
} // namespace

void RepairTopologyRender(state::Registry &r, mtl::ComputeChain &chain, std::span<const std::pair<state::Entity, const MeshTopologyEdit *>> edits, std::span<MeshletBuildSource> fresh) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<TriangleRepair> repairs;
    repairs.reserve(edits.size());
    for (const auto &[entity, topology] : edits) {
        const auto &record = meshes.Get(topology->StoreId);
        // Unchanged uniform keys can retain their meshlets unless their vertex representative was a corner retired by this topology edit.
        // Adding loose edges changes no surface vertices, corners, or normal sectors.
        const bool stable_keys = topology->Op == MeshTopologyOp::ExtrudeVertices || topology->Op == MeshTopologyOp::DeleteLoose ||
            (topology->Op == MeshTopologyOp::AddPrimitives && !topology->AddedTriangleCount) ||
            ((topology->Op == MeshTopologyOp::InsetRegion || topology->Op == MeshTopologyOp::InsetIndividual || TopologyClassifiesEdges(topology->Op)) &&
             !(record.CornerAttributes & MeshAttributeBit_Normal) &&
             topology->OriginalClassMode == record.Classification &&
             record.Classification != uint32_t(CornerClassMode::Mixed));
        // Patch planning reads the sources before the repair allocates chain scratch.
        repairs.push_back({entity, {
                                       .Changed = topology->ChangedTriangles.Triangles,
                                       .Removed = topology->Output->Replaced[1],
                                       .ReplacedCorners = topology->Output->Replaced[0],
                                       .Added = {topology->FirstTriangle, topology->AddedTriangleCount},
                                       .Sources = chain.Scratch.Get(topology->SourceTriangles),
                                       .StableKeys = stable_keys,
                                   }});
    }
    RepairTriangleRender(r, chain, repairs, fresh);
}

void RepairFaceRender(state::Registry &r, mtl::ComputeChain &chain, std::span<const std::pair<state::Entity, FaceTriangles>> changes) {
    std::vector<TriangleRepair> repairs;
    std::vector<state::Entity> entities;
    for (const auto &[entity, changed] : changes) {
        if (!changed.Count) continue;
        const auto &owner = RecordOf(r, entity);
        if (owner.ExtrasFaces.Count || !owner.PrimitiveData) throw std::invalid_argument("Local face repair requires published render ownership.");
        RefreshMeshBinding(r, owner.StoreId);
        repairs.push_back({entity, {.Changed = changed.Triangles}});
        entities.push_back(entity);
    }
    if (repairs.empty()) return;
    RepairTriangleRender(r, chain, repairs, {});
    RepointMeshInstances(r, entities);
}
