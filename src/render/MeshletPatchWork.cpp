#include "render/MeshletPatchWork.h"
#include "Profile.h"
#include "SortUnique.h"

#include "render/ElementWorkOps.h"
#include "render/GpuBufferOps.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletOwners.h"
#include "state/Registry.h"
#include <map>

namespace {
bool Contains(const BufferArena<uint32_t> &storage, ElementWork work, uint32_t element) {
    return work.Storage.Slot != InvalidSlot && ElementWorkRank(storage.Get(WorkStorageRange(work)), work, element) != InvalidOffset;
}
} // namespace

std::vector<uint32_t> ChangedClusters(state::Registry &r, const MeshStore::Record &owner, const BufferArena<uint32_t> &storage, ElementWork changed, ElementWork removed, ElementWork replaced_corners, bool stable_keys) {
    std::vector<uint32_t> clusters;
    if (changed.Storage.Slot == InvalidSlot) return clusters;
    const auto &render = r.Resources.get<const MeshStore>().Render();
    const auto owners = render.ElementMeshlets[0].View(owner.ElementMeshlets[0]);
    const auto records = render.Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto meshlet_entries = render.Meshlets.Entries(owner.MeshletData);
    const auto vertex_corners = render.MeshletVertexCorners.Get(owner.MeshletVertices.Values);
    ForEachWorkElement(storage, changed, [&](uint32_t triangle) {
        const auto cluster = owners[triangle];
        const auto &record = records[Resolve<RenderBlockSize>(meshlet_entries, ElementHandle{cluster}).Value()];
        const auto corners = vertex_corners.subspan(record.VertexOffset, record.VertexCount);
        if (!stable_keys || Contains(storage, removed, triangle) ||
            std::ranges::any_of(corners, [&](uint32_t corner) { return Contains(storage, replaced_corners, corner); })) {
            clusters.push_back(cluster);
        }
    });
    SortUnique(clusters);
    return clusters;
}

MeshletPatch PlanMeshletPatch(state::Registry &r, MeshStore::Record &owner, const BufferArena<uint32_t> &storage, const MeshletPatchInput &input) {
    const profile::CpuScope scope{"MeshletPatchPlan"};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto records = render.Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto meshlet_entries = render.Meshlets.Entries(owner.MeshletData);
    const auto record_of = [&](uint32_t cluster) -> const MeshletRecord & { return records[Resolve<RenderBlockSize>(meshlet_entries, ElementHandle{cluster}).Value()]; };
    const auto triangle_ids = render.MeshletTriangleIds.Get(owner.MeshletTriangles.Values);
    const auto triangle_entries = meshes.Arenas().Triangles.Entries(owner.TriangleData);
    MeshletPatch patch;
    // Grouped partitions precede ungrouped ones, each in ascending order.
    std::map<std::pair<bool, uint32_t>, MeshletPatchPartition> partitions;
    for (const auto cluster : input.Clusters) {
        const auto &record = record_of(cluster);
        const bool grouped = record.GroupIndex != InvalidOffset;
        partitions[{!grouped, grouped ? record.GroupIndex : record.Primitive}] = {.Group = record.GroupIndex, .Primitive = record.Primitive};
    }
    const auto place = [&](uint32_t triangle, uint32_t cluster) {
        if (cluster == InvalidOffset) {
            const auto address = Resolve<MeshElementBlockSize>(triangle_entries, ElementHandle{triangle}).Value();
            const auto &a = meshes.Arenas();
            const auto corner = a.Triangles.Get({address, 1u})[0].x;
            const auto face = a.HalfedgeFaces.Get({meshes.HandleAddress(owner.StoreId, MeshStore::ElementDomain::Halfedge, ElementHandle{corner}).Value(), 1u})[0];
            const auto material = a.FacePrimitives.View(owner.FacePrimitives).GetOr(face);
            const auto primitive = EnsureMeshletPrimitive(r, owner, material, 0u);
            const auto node_of = [&](uint32_t node) -> const LodNode & { return render.LodNodes.Get({render.LodNodes.Address(owner.NodeData, ElementHandle{node}).Value(), 1u})[0]; };
            auto leaf = render.Primitives.Get({render.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive}).Value(), 1u})[0].LodRootNode;
            while (node_of(leaf).ChildCount) leaf = node_of(leaf).ChildOffset;
            auto &partition = partitions[{true, primitive}];
            partition.Primitive = primitive;
            partition.Leaf = std::min(partition.Leaf, leaf);
            partition.Elements.push_back(triangle);
            return;
        }
        const auto &record = record_of(cluster);
        const bool grouped = record.GroupIndex != InvalidOffset;
        auto &partition = partitions.at({!grouped, grouped ? record.GroupIndex : record.Primitive});
        const auto leaf = render.MeshletLodLeaves.Get({Resolve<RenderBlockSize>(meshlet_entries, ElementHandle{cluster}).Value(), 1u})[0];
        partition.Leaf = std::min(partition.Leaf, leaf);
        partition.Elements.push_back(triangle);
    };
    for (const auto cluster : input.Clusters) {
        const auto &record = record_of(cluster);
        for (const auto triangle : triangle_ids.subspan(record.TriangleOffset, record.TriangleCount)) {
            if (!Contains(storage, input.Removed, triangle)) place(triangle, cluster);
        }
    }
    for (uint32_t i = 0u; const auto run : input.Added) {
        for (uint32_t t = 0u; t < run.Count; ++t) place(run.Offset + t, input.Sources[i++]);
    }
    for (auto &[key, partition] : partitions) {
        if (partition.Group != InvalidOffset) patch.Groups.push_back(partition.Group);
        if (!partition.Elements.empty()) patch.Partitions.push_back(std::move(partition));
    }
    return patch;
}

std::vector<uint32_t> AdoptMeshletFragments(state::Registry &r, mtl::ComputeChain &chain, MeshStore::Record &owner, uint32_t topology, std::span<MeshStore::Record> fragments, std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks) {
    const profile::CpuScope scope{"MeshletAdoptFragments"};
    auto &meshes = r.Resources.get<MeshStore>();
    auto &render = meshes.Render();
    std::vector<uint32_t> added;
    std::vector<Range> ranges;
    std::map<uint32_t, uint32_t> element_counts;
    const auto primitive_of = [&](uint32_t primitive) -> PrimitiveRecord & { return render.Primitives.GetMutable({render.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive}).Value(), 1u})[0]; };
    for (uint32_t p = 0u; p < fragments.size(); ++p) {
        const auto &job = jobs[p];
        for (uint32_t i = 0u; i < job.Count; ++i) added.push_back(job.First + i);
        ranges.push_back(fragments[p].Meshlets);
        if (topology != 0u) element_counts[job.Primitive] += job.ElementCount;
    }
    std::ranges::sort(added);
    if (added.empty()) return added;
    owner.Level0Count += uint32_t(added.size());
    ++owner.MeshletRevision;
    for (const auto &[primitive, count] : element_counts) primitive_of(primitive).TriangleCount += count;
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
