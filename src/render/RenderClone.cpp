#include "render/RenderClone.h"

#include "mesh/Mesh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshStores.h"
#include "metal/Dispatch.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletOwners.h"
#include "render/MeshletSpatial.h"
#include "render/SceneUpdates.h"
#include "state/Scene.h"

#include <algorithm>

namespace {
// A source's handles of one kind in ascending order, and the clone's handle of each at its rank.
// A render record references only its owner's handles of each kind, so every handle it maps is one of Sources.
struct HandleMap {
    std::vector<uint32_t> Sources;
    uint32_t First{InvalidOffset};
    uint32_t operator()(uint32_t handle) const {
        return handle == InvalidOffset ? InvalidOffset : First + uint32_t(std::ranges::lower_bound(Sources, handle) - Sources.begin());
    }
};

// The index visits a root's members in ascending order.
HandleMap Members(const MeshletIndex &index, uint32_t root) {
    HandleMap map;
    index.ForEach(root, [&](uint32_t id) { map.Sources.push_back(id); });
    return map;
}

// One clone's render owner, and the source maps that rebase its references.
struct Clone {
    const MeshBuffers *Source;
    MeshBuffers *Target;
    uint32_t SourceId, CloneId;
    HandleMap Meshlets, Primitives, Nodes, Groups;
};

// A membership root the batch's index update creates for a clone: a leaf node's members, or a dirty root's.
struct MemberRoot {
    enum class Kind : uint8_t { Leaf, PositionDirty, DirtyGroups };
    Clone *Owner;
    Kind Type;
    uint32_t Node{InvalidOffset};
    std::vector<uint32_t> Members;
};

// Packs the source's cluster payloads in cluster order on `copies`, so a source built in one pass keeps its construction layout and copies in a few runs.
// Its clusters and traversal leaves copy with every reference rebased.
void CopyClusters(GpuBuffers &buffers, const MeshStore &meshes, CloneCopies &copies, Clone &clone) {
    const auto &source = *clone.Source;
    auto &target = *clone.Target;
    const auto source_records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    uint64_t triangle_ids = 0u, vertex_corners = 0u, local_triangles = 0u;
    for (const auto id : clone.Meshlets.Sources) {
        const auto &record = source_records[id];
        if (record.RefinedGroup == InvalidOffset) triangle_ids += record.TriangleCount;
        vertex_corners += record.VertexCount;
        if (record.Topology == 0u) local_triangles += uint64_t(record.TriangleCount) * 3u;
    }
    target.MeshletTriangles = buffers.MeshletTriangleIds.Allocate(uint32_t(triangle_ids));
    target.MeshletVertices = buffers.MeshletVertexCorners.Allocate(uint32_t(vertex_corners));
    target.MeshletLocalTriangles = buffers.MeshletLocalTriangles.Allocate(uint32_t(local_triangles));
    buffers.MeshletTriangleIds.CaptureWrite(target.MeshletTriangles);
    buffers.MeshletVertexCorners.CaptureWrite(target.MeshletVertices);
    buffers.MeshletLocalTriangles.CaptureWrite(target.MeshletLocalTriangles);
    const auto copy = [&](auto &arena, uint64_t from, uint64_t to, uint64_t count) {
        constexpr uint64_t Size = sizeof(typename decltype(arena.Get(Range{}))::element_type);
        copies.Copy(arena.Buffer, from * Size, to * Size, count * Size);
    };
    auto records = buffers.Meshlets.GetMutable(target.Meshlets);
    auto leaves = buffers.MeshletLodLeaves.GetMutable(target.Meshlets);
    const auto source_leaves = buffers.MeshletLodLeaves.Buffer.GetSpan<uint32_t>();
    uint32_t next_triangle = target.MeshletTriangles.Offset, next_vertex = target.MeshletVertices.Offset, next_local = target.MeshletLocalTriangles.Offset;
    for (uint32_t m = 0u; m < clone.Meshlets.Sources.size(); ++m) {
        const auto id = clone.Meshlets.Sources[m];
        auto record = source_records[id];
        if (record.RefinedGroup == InvalidOffset) {
            copy(buffers.MeshletTriangleIds, record.TriangleOffset, next_triangle, record.TriangleCount);
            record.TriangleOffset = std::exchange(next_triangle, next_triangle + record.TriangleCount);
        }
        copy(buffers.MeshletVertexCorners, record.VertexOffset, next_vertex, record.VertexCount);
        record.VertexOffset = std::exchange(next_vertex, next_vertex + record.VertexCount);
        if (record.Topology == 0u) {
            copy(buffers.MeshletLocalTriangles, record.LocalTriangleOffset, next_local, uint64_t(record.TriangleCount) * 3u);
            record.LocalTriangleOffset = std::exchange(next_local, next_local + record.TriangleCount * 3u);
        }
        record.Primitive = clone.Primitives(record.Primitive);
        record.GroupIndex = clone.Groups(record.GroupIndex);
        record.RefinedGroup = clone.Groups(record.RefinedGroup);
        records[m] = record;
        leaves[m] = clone.Nodes(source_leaves[id]);
    }
    // Triangle clusters name canonical triangles and corners, which the clone holds at a fixed offset from its source's.
    // Point and line clusters name elements and vertices relative to their owner's origin.
    if (source.RenderTopology != 0u) return;
    const auto &a = meshes.Arenas();
    const auto &source_record = meshes.Get(clone.SourceId), &clone_record = meshes.Get(clone.CloneId);
    copies.Rebase(buffers.MeshletTriangleIds.Buffer, uint64_t(target.MeshletTriangles.Offset) * sizeof(uint32_t), target.MeshletTriangles.Count,
                  a.Triangles.First(clone_record.TriangleData) - a.Triangles.First(source_record.TriangleData));
    copies.Rebase(buffers.MeshletVertexCorners.Buffer, uint64_t(target.MeshletVertices.Offset) * sizeof(uint32_t), target.MeshletVertices.Count,
                  a.FaceCorners.First(clone_record.FaceCorners) - a.FaceCorners.First(source_record.FaceCorners));
}

// Copies the source's primitives and their routes.
// A primitive's construction slice keeps its place in the clone's packed triangle IDs.
void CopyPrimitives(GpuBuffers &buffers, const MeshStore &meshes, const Clone &clone) {
    const auto &source = *clone.Source;
    auto &target = *clone.Target;
    const auto materials = OffsetOrInvalid(meshes.Get(clone.CloneId).PrimitiveMaterials);
    auto primitives = buffers.Primitives.GetMutable(target.Primitives);
    for (uint32_t p = 0u; p < clone.Primitives.Sources.size(); ++p) {
        auto primitive = buffers.Primitives.Get({clone.Primitives.Sources[p], 1u})[0];
        primitive.PrimitiveMaterialOffset = materials;
        primitive.TriangleOffset = primitive.TriangleOffset - source.MeshletTriangles.Offset + target.MeshletTriangles.Offset;
        primitive.LodRootNode = clone.Nodes(primitive.LodRootNode);
        primitive.LodFinestNode = clone.Nodes(primitive.LodFinestNode);
        primitives[p] = primitive;
    }
    if (!source.PrimitiveRoutes.Count) return;
    std::vector<uint32_t> routes;
    for (const auto route : buffers.PrimitiveRoutes.Get(source.PrimitiveRoutes)) routes.push_back(clone.Primitives(route));
    target.PrimitiveRoutes = buffers.PrimitiveRoutes.Allocate(routes);
}

// Copies the source's LOD nodes and parents, and lists each leaf's members for a membership root of its own.
void CopyNodes(GpuBuffers &buffers, Clone &clone, std::vector<MemberRoot> &roots) {
    auto &target = *clone.Target;
    auto nodes = buffers.LodNodes.GetMutable(target.LodNodes);
    auto parents = buffers.LodParents.GetMutable(target.LodNodes);
    for (uint32_t n = 0u; n < clone.Nodes.Sources.size(); ++n) {
        const auto id = clone.Nodes.Sources[n];
        auto node = buffers.LodNodes.Get({id, 1u})[0];
        if (node.ChildCount) node.ChildOffset = clone.Nodes(node.ChildOffset);
        if (node.MeshletRoot != InvalidOffset) {
            auto &leaf = roots.emplace_back(MemberRoot{.Owner = &clone, .Type = MemberRoot::Kind::Leaf, .Node = target.LodNodes.Offset + n});
            buffers.ActiveMeshlets.ForEach(node.MeshletRoot, [&](uint32_t member) { leaf.Members.push_back(clone.Meshlets(member)); });
        }
        node.MeshletRoot = InvalidOffset;
        nodes[n] = node;
        parents[n] = clone.Nodes(buffers.LodParents.Get({id, 1u})[0]);
    }
}

// Copies the source's cluster groups, with each group's member run then proxy run filling one allocation.
void CopyGroups(GpuBuffers &buffers, const Clone &clone) {
    uint64_t group_ids = 0u;
    for (const auto group : clone.Groups.Sources) {
        const auto &links = buffers.GroupLinks.Get({group, 1u})[0];
        group_ids += links.MemberCount + links.ProxyCount;
    }
    const auto runs = buffers.GroupClusterIds.Allocate(uint32_t(group_ids));
    auto ids = buffers.GroupClusterIds.GetMutable(runs);
    auto groups = buffers.ClusterGroups.GetMutable(clone.Target->ClusterGroups);
    auto links = buffers.GroupLinks.GetMutable(clone.Target->ClusterGroups);
    uint32_t next = 0u;
    const auto copy_run = [&](uint32_t offset, uint32_t count) {
        const auto first = runs.Offset + next;
        for (const auto id : buffers.GroupClusterIds.Get({offset, count})) ids[next++] = clone.Meshlets(id);
        return first;
    };
    for (uint32_t g = 0u; g < clone.Groups.Sources.size(); ++g) {
        const auto group = clone.Groups.Sources[g];
        groups[g] = buffers.ClusterGroups.Get({group, 1u})[0];
        const auto source_links = buffers.GroupLinks.Get({group, 1u})[0];
        const auto member_offset = copy_run(source_links.MemberOffset, source_links.MemberCount);
        links[g] = {.MemberOffset = member_offset, .MemberCount = source_links.MemberCount,
                    .ProxyOffset = copy_run(source_links.ProxyOffset, source_links.ProxyCount), .ProxyCount = source_links.ProxyCount};
    }
}

// Copies the owners of the source's element blocks onto the clone's blocks, naming the clone's clusters.
// A clone's elements sit at its source's offsets within its own domain, so each owner block moves by whole blocks.
void CopyElementOwners(GpuBuffers &buffers, const MeshStore &meshes, const Clone &clone) {
    const auto &source = *clone.Source;
    auto &target = *clone.Target;
    if (!source.ElementMeshletBlockCount) return;
    const auto topology = source.RenderTopology;
    auto &owners = buffers.ElementMeshlets[topology];
    const auto source_blocks = buffers.MeshletOwnerBlocks(source);
    const auto clone_first = ElementDomainFirst(meshes, clone.CloneId, topology);
    const auto block_delta = int64_t(clone_first / MeshElementBlockSize) - int64_t(ElementDomainFirst(meshes, clone.SourceId, topology) / MeshElementBlockSize);
    std::vector<uint32_t> target_blocks;
    for (const auto block : source_blocks) target_blocks.push_back(uint32_t(int64_t(block) + block_delta));
    owners.Attach(target_blocks, InvalidOffset);
    for (uint32_t b = 0u; b < source_blocks.size(); ++b) {
        const auto values = owners.Values.Buffer.GetSpan<uint32_t>(owners.Payload(source_blocks[b] * MeshElementBlockSize, MeshElementBlockSize));
        const auto to = owners.Edit(target_blocks[b] * MeshElementBlockSize, MeshElementBlockSize);
        for (uint32_t e = 0u; e < MeshElementBlockSize; ++e) to[e] = clone.Meshlets(values[e]);
    }
    target.ElementMeshletBlockCount = uint32_t(source_blocks.size());
    // Triangle owners name absolute triangle handles, and point and line owners name elements from the domain's first handle.
    target.ElementMeshletOrigin = topology == 0u ? 0u : clone_first;
}
} // namespace

void CloneRenderRecords(state::Scene &r, mtl::ComputeChain &chain, CloneCopies &copies, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    auto &index = buffers.ActiveMeshlets;
    // Every clone's render owner exists before any of them is referenced, since emplacing one can move the others.
    std::vector<uint32_t> cloned;
    for (uint32_t i = 0u; i < source_ids.size(); ++i) {
        const auto *source = buffers.TryMeshOf(source_ids[i]);
        if (!source || source->MeshletRoot == InvalidOffset || buffers.TryMeshOf(clone_ids[i])) continue;
        const auto set = meshes.Get(clone_ids[i]).Vertices;
        buffers.EmplaceMesh(clone_ids[i], {{a.Vertices.First(set), a.Vertices.Count(set)}, a.Vertices.Buffer.Slot});
        cloned.push_back(i);
    }
    if (cloned.empty()) return;
    std::vector<Clone> clones;
    clones.reserve(cloned.size());
    uint64_t meshlet_count = 0u, primitive_count = 0u, node_count = 0u, group_count = 0u;
    for (const auto i : cloned) {
        auto &clone = clones.emplace_back(Clone{.Source = &buffers.MeshOf(source_ids[i]), .Target = &buffers.MeshOf(clone_ids[i]), .SourceId = source_ids[i], .CloneId = clone_ids[i]});
        clone.Meshlets = Members(index, clone.Source->MeshletRoot);
        clone.Primitives = Members(index, clone.Source->PrimitiveRoot);
        clone.Nodes = Members(index, clone.Source->NodeRoot);
        clone.Groups = Members(index, clone.Source->GroupRoot);
        meshlet_count += clone.Meshlets.Sources.size();
        primitive_count += clone.Primitives.Sources.size();
        node_count += clone.Nodes.Sources.size();
        group_count += clone.Groups.Sources.size();
    }
    // Every arena grows once for the batch.
    buffers.Meshlets.ReserveAdditional(meshlet_count);
    buffers.MeshletLodLeaves.ReserveAdditional(meshlet_count);
    buffers.MeshletSpatialNodes.ReserveAdditional(meshlet_count);
    buffers.Primitives.ReserveAdditional(primitive_count);
    buffers.LodNodes.ReserveAdditional(node_count);
    buffers.LodParents.ReserveAdditional(node_count);
    buffers.ClusterGroups.ReserveAdditional(group_count);
    buffers.GroupLinks.ReserveAdditional(group_count);
    buffers.MeshRecords.ReserveAdditional(clones.size());
    std::vector<MeshletIndexEdit> ownership;
    std::vector<MemberRoot> roots;
    for (auto &clone : clones) {
        const auto &source = *clone.Source;
        auto &target = *clone.Target;
        target.StoreId = clone.CloneId;
        target.RenderTopology = source.RenderTopology;
        target.Level0Count = source.Level0Count;
        target.MeshletRevision = source.MeshletRevision;
        AssignFaceIndices(meshes, Mesh{meshes, clone.CloneId}, target);
        target.Meshlets = buffers.AllocateMeshlets(uint32_t(clone.Meshlets.Sources.size()));
        clone.Meshlets.First = target.Meshlets.Offset;
        target.Primitives = buffers.Primitives.Allocate(uint32_t(clone.Primitives.Sources.size()));
        clone.Primitives.First = target.Primitives.Offset;
        target.LodNodes = buffers.LodNodes.Allocate(uint32_t(clone.Nodes.Sources.size()));
        buffers.LodParents.Mirror(target.LodNodes);
        clone.Nodes.First = target.LodNodes.Offset;
        target.ClusterGroups = buffers.ClusterGroups.Allocate(uint32_t(clone.Groups.Sources.size()));
        buffers.GroupLinks.Mirror(target.ClusterGroups);
        clone.Groups.First = target.ClusterGroups.Offset;
        CopyClusters(buffers, meshes, copies, clone);
        CopyPrimitives(buffers, meshes, clone);
        CopyNodes(buffers, clone, roots);
        CopyGroups(buffers, clone);
        CopyElementOwners(buffers, meshes, clone);
        const auto mesh_record = BuildMeshRecord(buffers, target, meshes, clone.CloneId, target.RenderTopology == 0u, target.RenderTopology == 1u);
        target.MeshRecord = buffers.MeshRecords.Allocate(std::span{&mesh_record, 1u});
        for (const auto range : {target.Meshlets, target.Primitives, target.LodNodes, target.ClusterGroups}) ownership.push_back({.Insert = range});
        for (const auto [root, kind] : {std::pair{source.PositionDirtyRoot, MemberRoot::Kind::PositionDirty}, std::pair{source.DirtyGroupRoot, MemberRoot::Kind::DirtyGroups}}) {
            if (root == InvalidOffset) continue;
            auto &dirty = roots.emplace_back(MemberRoot{.Owner = &clone, .Type = kind});
            const auto &map = kind == MemberRoot::Kind::PositionDirty ? clone.Meshlets : clone.Groups;
            index.ForEach(root, [&](uint32_t member) { dirty.Members.push_back(map(member)); });
        }
    }
    const auto first_root = uint32_t(ownership.size());
    for (const auto &root : roots) ownership.push_back({.Added = root.Members});
    index.Update(ownership);
    for (uint32_t i = 0u; i < roots.size(); ++i) {
        const auto &[owner, kind, node, _] = roots[i];
        const auto root = ownership[first_root + i].Root;
        switch (kind) {
            case MemberRoot::Kind::Leaf: buffers.LodNodes.GetMutable({node, 1u})[0].MeshletRoot = root; break;
            case MemberRoot::Kind::PositionDirty: owner->Target->PositionDirtyRoot = root; break;
            case MemberRoot::Kind::DirtyGroups: owner->Target->DirtyGroupRoot = root; break;
        }
    }
    auto &scene = r.Context.get<GpuSceneState>();
    std::vector<state::Entity> entities;
    for (uint32_t c = 0u; c < clones.size(); ++c) {
        auto &target = *clones[c].Target;
        const auto *edits = &ownership[4u * c];
        target.MeshletRoot = edits[0].Root;
        target.PrimitiveRoot = edits[1].Root;
        target.NodeRoot = edits[2].Root;
        target.GroupRoot = edits[3].Root;
        const auto entity = MeshEntityOf(r, clones[c].CloneId);
        entities.push_back(entity);
        if (target.PositionDirtyRoot != InvalidOffset) scene.PositionDirty.insert(entity);
        if (target.DirtyGroupRoot != InvalidOffset) scene.LodDirty.insert(entity);
        if (!buffers.ClusterGroupCount(target)) scene.LodDemand.insert(entity);
    }
    RepointMeshInstances(r, entities);
    buffers.PreludeStale = true;
    // A spatial tree reads the clone's geometry, which the chain's submit copies.
    std::vector<uint32_t> spatial;
    for (const auto &clone : clones)
        if (clone.Source->SpatialRoot != InvalidOffset) spatial.push_back(clone.CloneId);
    if (spatial.empty()) return;
    chain.AfterSubmit([&r, spatial = std::move(spatial)] {
        auto &buffers = r.Context.get<GpuBuffers>();
        std::vector<MeshBuffers *> owners;
        for (const auto id : spatial) owners.push_back(&buffers.MeshOf(id));
        BuildMeshletSpatial(r, owners);
    });
}
