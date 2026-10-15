#include "render/MeshletPatchWork.h"
#include "Profile.h"
#include "SortUnique.h"

#include "mesh/MeshStore.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBufferOps.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletOwners.h"
#include "state/Scene.h"
#include <map>

MeshletPatch PlanMeshletPatch(state::Scene &r, MeshStore::Record &owner, const BufferArena<uint32_t> &storage, const MeshletPatchInput &input) {
    const profile::CpuScope scope{"MeshletPatchPlan"};
    if (input.Sources.size() != input.Added.Count || uint64_t(input.Added.Offset) + input.Added.Count > UINT32_MAX) {
        throw std::invalid_argument("Meshlet patch provenance must cover the emitted triangle run.");
    }
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto owners = render.ElementMeshlets[0].View(owner.ElementMeshlets[0]);
    const auto records = render.Meshlets.Buffer.GetSpan<MeshletRecord>();
    // The owner's cluster at a meshlet handle.
    const auto meshlet_entries = render.Meshlets.Entries(owner.MeshletData);
    const auto record_of = [&](uint32_t cluster) -> const MeshletRecord & { return records[Resolve<RenderBlockSize>(meshlet_entries, ElementHandle{cluster}).Value()]; };
    const auto triangle_ids = render.MeshletTriangleIds.Get(owner.MeshletTriangles.Values);
    const auto contains = [&](ElementWork work, uint32_t element) {
        return work.Storage.Slot != InvalidSlot && ElementWorkRank(storage.Get(WorkStorageRange(work)), work, element) != InvalidOffset;
    };
    const auto triangle_entries = meshes.Arenas().Triangles.Entries(owner.TriangleData);
    const auto present = [&](uint32_t triangle) {
        return triangle / MeshElementBlockSize < triangle_entries.size() && triangle_entries[triangle / MeshElementBlockSize] != InvalidOffset;
    };
    // The live finest cluster whose payload holds the triangle.
    const auto cluster_of = [&](uint32_t triangle) {
        const auto cluster = present(triangle) ? owners.GetOr(triangle, InvalidOffset) : InvalidOffset;
        if (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{cluster})) throw std::invalid_argument("Meshlet patch triangle has no live render owner.");
        const auto &record = record_of(cluster);
        if (record.Topology || record.RefinedGroup != InvalidOffset || record.TriangleCount > 48u ||
            !std::ranges::contains(triangle_ids.subspan(record.TriangleOffset, record.TriangleCount), triangle)) {
            throw std::invalid_argument("Meshlet patch triangle disagrees with its render owner.");
        }
        return cluster;
    };
    MeshletPatch patch;
    if (input.Changed.Storage.Slot != InvalidSlot) {
        ForEachWorkElement(storage, input.Changed, [&](uint32_t triangle) {
            const auto cluster = cluster_of(triangle);
            const auto &record = record_of(cluster);
            const auto corners = render.MeshletVertexCorners.Get(owner.MeshletVertices.Values).subspan(record.VertexOffset, record.VertexCount);
            if (!input.StableKeys || contains(input.Removed, triangle) ||
                std::ranges::any_of(corners, [&](uint32_t corner) { return contains(input.ReplacedCorners, corner); })) {
                patch.Clusters.push_back(cluster);
            }
        });
    }
    SortUnique(patch.Clusters);
    // Grouped partitions precede ungrouped ones, each in ascending order.
    std::map<std::pair<bool, uint32_t>, MeshletPatchPartition> partitions;
    for (const auto cluster : patch.Clusters) {
        const auto &record = record_of(cluster);
        if (!render.Primitives.Contains(owner.PrimitiveData, ElementHandle{record.Primitive}) ||
            (record.GroupIndex != InvalidOffset && !render.ClusterGroups.Contains(owner.GroupData, ElementHandle{record.GroupIndex}))) {
            throw std::invalid_argument("Meshlet patch cluster names a foreign primitive or group.");
        }
        const bool grouped = record.GroupIndex != InvalidOffset;
        partitions[{!grouped, grouped ? record.GroupIndex : record.Primitive}] = {.Group = record.GroupIndex, .Primitive = record.Primitive};
    }
    // Every partition is spatially local to one existing simplification group.
    // It takes a deterministic source leaf, and span repair redistributes overflow.
    const auto place = [&](uint32_t triangle, uint32_t source) {
        if (!meshes.Arenas().Triangles.Contains(triangle_entries, ElementHandle{triangle})) throw std::invalid_argument("Meshlet patch triangle is not live in its mesh.");
        const auto address = Resolve<MeshElementBlockSize>(triangle_entries, ElementHandle{triangle}).Value();
        if (source == InvalidOffset) {
            const auto &a = meshes.Arenas();
            const auto corner = a.Triangles.Get({address, 1u})[0].x;
            const auto face = a.HalfedgeFaces.Get({meshes.HandleAddress(owner.StoreId, MeshStore::ElementDomain::Halfedge, ElementHandle{corner}).Value(), 1u})[0];
            const auto material = a.FacePrimitives.View(owner.FacePrimitives).GetOr(face);
            const auto primitive = EnsureMeshletPrimitive(r, owner, material, 0u);
            // An ungrouped face joins an actual traversal leaf, even when the
            // existing primitive has a separate pinned finest node for LOD.
            const auto node_of = [&](uint32_t node) -> const LodNode & { return render.LodNodes.Get({render.LodNodes.Address(owner.NodeData, ElementHandle{node}).Value(), 1u})[0]; };
            auto leaf = render.Primitives.Get({render.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive}).Value(), 1u})[0].LodRootNode;
            while (node_of(leaf).ChildCount) leaf = node_of(leaf).ChildOffset;
            auto &partition = partitions[{true, primitive}];
            partition.Primitive = primitive;
            partition.Leaf = std::min(partition.Leaf, leaf);
            partition.Elements.push_back(triangle);
            return;
        }
        const auto cluster = cluster_of(source);
        if (!std::ranges::binary_search(patch.Clusters, cluster)) throw std::invalid_argument("Meshlet patch triangle keeps its source cluster.");
        const auto &record = record_of(cluster);
        const bool grouped = record.GroupIndex != InvalidOffset;
        auto &partition = partitions.at({!grouped, grouped ? record.GroupIndex : record.Primitive});
        const auto leaf = render.MeshletLodLeaves.Get({Resolve<RenderBlockSize>(meshlet_entries, ElementHandle{cluster}).Value(), 1u})[0];
        bool held = false;
        if (render.LodNodes.Contains(owner.NodeData, ElementHandle{leaf})) {
            const auto &node = render.LodNodes.Get({render.LodNodes.Address(owner.NodeData, ElementHandle{leaf}).Value(), 1u})[0];
            if (!node.ChildCount) meshes.ForEachLodMember(owner, node, [&](uint32_t member) { held |= member == cluster; });
        }
        if (!held) throw std::invalid_argument("Meshlet patch cluster disagrees with its traversal leaf.");
        partition.Leaf = std::min(partition.Leaf, leaf);
        partition.Elements.push_back(triangle);
    };
    for (const auto cluster : patch.Clusters) {
        const auto &record = record_of(cluster);
        for (const auto triangle : triangle_ids.subspan(record.TriangleOffset, record.TriangleCount)) {
            if (!contains(input.Removed, triangle)) place(triangle, triangle);
        }
    }
    for (uint32_t i = 0u; i < input.Added.Count; ++i) place(input.Added.Offset + i, input.Sources[i]);
    for (auto &[key, partition] : partitions) {
        if (partition.Group != InvalidOffset) patch.Groups.push_back(partition.Group);
        if (!partition.Elements.empty()) patch.Partitions.push_back(std::move(partition));
    }
    return patch;
}

std::vector<uint32_t> AdoptMeshletFragments(state::Scene &r, mtl::ComputeChain &chain, MeshStore::Record &owner, uint32_t topology, std::span<MeshStore::Record> fragments, std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks) {
    const profile::CpuScope scope{"MeshletAdoptFragments"};
    if (topology > 2u || fragments.size() != jobs.size() || !owner.PrimitiveData || !owner.NodeData) {
        throw std::invalid_argument("Meshlet fragments require one matching canonical owner and job each.");
    }
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    std::vector<uint32_t> added;
    std::vector<Range> ranges;
    std::map<uint32_t, uint64_t> element_counts;
    const auto primitive_of = [&](uint32_t primitive) -> PrimitiveRecord & { return render.Primitives.GetMutable({render.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive}).Value(), 1u})[0]; };
    for (uint32_t p = 0u; p < fragments.size(); ++p) {
        const auto &fragment = fragments[p];
        const auto &job = jobs[p];
        if (&fragment == &owner || fragment.StoreId != owner.StoreId || fragment.MeshletData || fragment.PrimitiveData || fragment.NodeData || fragment.GroupData || fragment.MemberData ||
            fragment.PrimitiveRoutes.Count || fragment.RenderTopologies != 0u || std::ranges::any_of(fragment.ElementMeshletBlockCounts, [](uint32_t count) { return count != 0u; }) ||
            fragment.Level0Count != fragment.Meshlets.Count ||
            job.First != fragment.Meshlets.Offset || job.Count != fragment.Meshlets.Count ||
            !render.Primitives.Contains(owner.PrimitiveData, ElementHandle{job.Primitive}) ||
            render.Primitives.Get({render.Primitives.Address(owner.PrimitiveData, ElementHandle{job.Primitive}).Value(), 1u})[0].Topology != topology ||
            (job.Group != InvalidOffset && !render.ClusterGroups.Contains(owner.GroupData, ElementHandle{job.Group})) ||
            (job.Count && (!render.LodNodes.Contains(owner.NodeData, ElementHandle{job.Leaf}) || render.LodNodes.Get({render.LodNodes.Address(owner.NodeData, ElementHandle{job.Leaf}).Value(), 1u})[0].ChildCount))) {
            throw std::invalid_argument("Meshlet adoption requires independently owned, unpublished fragments.");
        }
        for (uint32_t i = 0u; i < job.Count; ++i) {
            if (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{job.First + i})) throw std::invalid_argument("Meshlet adoption names a cluster outside its owner.");
            added.push_back(job.First + i);
        }
        ranges.push_back(fragment.Meshlets);
        if (topology != 0u) element_counts[job.Primitive] += job.ElementCount;
    }
    if (added.size() > UINT32_MAX - owner.Level0Count) throw std::length_error("Meshlet adoption exceeds canonical cluster capacity.");
    for (const auto &[primitive, count] : element_counts) {
        if (count > UINT32_MAX - primitive_of(primitive).TriangleCount) throw std::length_error("Element adoption exceeds primitive element capacity.");
    }
    std::ranges::sort(added);
    if (added.empty()) return added;
    if (std::ranges::adjacent_find(added) != added.end()) throw std::invalid_argument("Meshlet adoption repeats an owned cluster.");
    owner.Level0Count += uint32_t(added.size());
    ++owner.MeshletRevision;
    for (const auto &[primitive, count] : element_counts) primitive_of(primitive).TriangleCount += uint32_t(count);
    // A fragment's clusters are a run of handles and of addresses alike.
    std::vector<uint32_t> addresses;
    addresses.reserve(added.size());
    for (const auto &job : jobs) {
        if (!job.Count) continue;
        const auto first = render.Meshlets.Address(owner.MeshletData, ElementHandle{job.First}).Value();
        std::ranges::fill(render.MeshletLodLeaves.GetMutable({first, job.Count}), job.Leaf);
        for (uint32_t i = 0u; i < job.Count; ++i) addresses.push_back(first + i);
    }
    std::ranges::sort(addresses);
    // The owner now owns every new payload, so exception cleanup finds no provisional range to release.
    for (auto &fragment : fragments) fragment = {};
    std::vector<MeshletOwnersJob> owner_jobs;
    PrepareMeshletOwners(r, owner, topology, ranges, blocks, owner_jobs);
    RecordMeshletOwners(r, chain, owner_jobs);
    UpdatePosedMeshletBlocks(r, owner, addresses);
    return added;
}
