#include "render/GpuBuffers.h"
#include "render/MeshletBuild.h"
#include "render/MeshletBuildGpu.h"
#include "state/Scene.h"

#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include <utility>

// The mesh's shared arena locations and table runs.
// An extras record draws its bone or joint faces with the instance transforms, naming its vertex handles through its vertex table.
MeshRecord BuildMeshRecord(const GpuBuffers &buffers, const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    const auto &arenas = meshes.Arenas();
    const auto &render = meshes.Render();
    const auto vertex_count = arenas.Vertices.Count(record.Vertices);
    if (record.ExtrasFaces.Count) {
        return {
            .VertexSlot = arenas.Vertices.Buffer.Slot,
            .IndexSlotOffset = {render.ExtrasFaces.Buffer.Slot, record.ExtrasFaces.Offset},
            .ModelSlot = buffers.Instances.TransformBuffer.Slot,
            .Connectivity = {.Tables = meshes.GetElementTables(store_id)},
            .VertexCountOrHeadImageSlot = vertex_count,
        };
    }
    return {
        .VertexSlot = arenas.Vertices.Buffer.Slot,
        .IndexSlotOffset = {arenas.FaceCorners.Buffer.Slot, 0u},
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .TriangleSlot = arenas.FaceTriangles.Count(record.FaceData) ? arenas.Triangles.Buffer.Slot : InvalidSlot,
        .CornerClassMode = meshes.GetCornerClassMode(store_id),
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .HalfedgeCount = arenas.FaceCorners.Count(record.FaceCorners),
        .FaceCount = arenas.FaceTriangles.Count(record.FaceData),
        .VertexCountOrHeadImageSlot = vertex_count,
        .MorphShadingAuthored = record.MorphShadingAuthored ? 1u : 0u,
        .PrimitiveMaterialOffset = OffsetOrInvalid(record.PrimitiveMaterials),
        .Layers = meshes.Layers(store_id),
    };
}

void RefreshMeshBinding(state::Scene &r, uint32_t store_id) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto *record = meshes.TryGet(store_id);
    if (!record || record->RenderTopologies == 0u) return;
    meshes.Render().MeshRecords.GetMutable({store_id, 1u})[0] = BuildMeshRecord(r.Context.get<const GpuBuffers>(), meshes, store_id);
}

MeshletBuildInputs CaptureMeshletInputs(const Mesh &mesh, const MeshStore &meshes, TriangleCorners triangle_corners) {
    const uint32_t store_id = mesh.GetStoreId();

    const bool face_topology = mesh.FaceCount() > 0u;
    const auto &record = meshes.Get(store_id);
    const auto &arenas = meshes.Arenas();
    MeshletBuildInputs inputs{
        .Indices = {triangle_corners, arenas.FaceCorners.Buffer.GetSpan<uint32_t>()},
        .Vertices = arenas.Vertices.Buffer.GetSpan<Vertex>(),
        .VertexTable = arenas.Vertices.Entries(record.Vertices),
        .DenseVertices = record.Vertices && (arenas.Vertices.Set(record.Vertices).Flags & 1u) ? Range{0u, arenas.Vertices.Count(record.Vertices)} : Range{},
        .Normals = meshes.GetCornerNormalView(store_id),
        .Weld = {
            .CornerClassMode = meshes.GetCornerClassMode(store_id),
            .CornerSectors = {arenas.CornerSectors.View(record.CornerSectors), triangle_corners},
            .FaceSharpness = arenas.FaceSharpness.Buffer.GetSpan<uint8_t>(),
            .FaceTable = arenas.FaceTriangles.Entries(record.FaceData),
            .TriangleFaces = {triangle_corners, arenas.HalfedgeFaces.Buffer.GetSpan<uint32_t>()},
            .CustomNormals = {arenas.CustomNormals.View(record.CustomNormals), triangle_corners},
            .CornerUvs = {
                CornerAttributeView<vec2>{arenas.CornerUvs[0].View(record.CornerUvs[0]), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[1].View(record.CornerUvs[1]), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[2].View(record.CornerUvs[2]), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[3].View(record.CornerUvs[3]), triangle_corners},
            },
            .CornerTangents = CornerAttributeView<vec4>{arenas.CornerTangents.View(record.CornerTangents), triangle_corners},
            .CornerColors = CornerAttributeView<vec4>{arenas.CornerColors.View(record.CornerColors), triangle_corners},
            .MorphShadingAuthored = meshes.Get(store_id).MorphShadingAuthored,
        },
        .FaceTopology = face_topology,
    };
    return inputs;
}

ClusterLodBuild BuildMeshletClusterLod(const MeshStore &meshes, const MeshStore::Record &mb, const MeshletBuildInputs &in, std::span<const uint32_t> primitive_triangle_counts) {
    const auto &render = meshes.Render();
    std::vector<PrimitiveRecord> placed;
    meshes.ForEachPrimitive(mb, [&](ElementHandle, const PrimitiveRecord &primitive) { if (primitive.Topology==0u) placed.push_back(primitive); });
    if (!ClusterLodApplies(in.FaceTopology, mb.Level0Count)) return {};
    assert(meshes.ClusterGroupCount(mb) == 0u);
    std::vector<ClusterLodPrimitive> primitives;
    std::vector<ClusterLodSourceCluster> clusters;
    clusters.reserve(mb.Level0Count);
    if (!primitive_triangle_counts.empty() && primitive_triangle_counts.size() != placed.size()) {
        throw std::invalid_argument("Live LOD primitive triangle counts do not match the owner.");
    }
    const auto meshlets = render.Meshlets.Entries(mb.MeshletData);
    uint32_t triangle_cursor = 0u;
    for (uint32_t p = 0u; p < placed.size(); ++p) {
        const auto &primitive = placed[p];
        const uint32_t first_triangle = primitive_triangle_counts.empty() ? primitive.TriangleOffset : triangle_cursor;
        const uint32_t triangle_count = primitive_triangle_counts.empty() ? primitive.TriangleCount : primitive_triangle_counts[p];
        if (!primitive_triangle_counts.empty()) triangle_cursor += triangle_count;
        const auto first_cluster = uint32_t(clusters.size());
        if (primitive.LodFinestNode != InvalidOffset) {
            meshes.ForEachLodMember(mb, render.LodNodes.Get({render.LodNodes.Address(mb.NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0], [&](uint32_t id) {
                const auto &record = render.Meshlets.Get({Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value(), 1u})[0];
                clusters.push_back({
                    .FirstVertex = record.VertexOffset,
                    .VertexCount = record.VertexCount,
                    .FirstLocalTriangle = record.LocalTriangleOffset,
                    .TriangleCount = record.TriangleCount,
                    .Center = record.Center,
                    .Radius = record.Radius,
                    .ConeCullSafe = (record.ConeAxisCutoff >> 24u) != 127u,
                });
            });
        }
        primitives.push_back({
            .FirstTriangle = first_triangle,
            .TriangleCount = triangle_count,
            .FirstCluster = first_cluster,
            .ClusterCount = uint32_t(clusters.size()) - first_cluster,
            .Attributes = primitive.LodAttributes,
        });
    }
    if (!primitive_triangle_counts.empty() && uint64_t(triangle_cursor) * 3u != in.Indices.size()) {
        throw std::invalid_argument("Live LOD triangle handles do not cover the primitive inputs.");
    }
    return BuildClusterLod(ClusterLodMesh{
        .CornerVertices = in.Indices,
        .Positions = &in.Vertices.front().Position.x,
        .PositionStride = sizeof(Vertex),
        .VertexTable = in.VertexTable,
        .DenseVertices = in.DenseVertices,
        .Normals = in.Normals,
        .Weld = in.Weld,
        .Primitives = primitives,
        .Clusters = clusters,
        .SourceVertexCorners = render.MeshletVertexCorners.Get(mb.MeshletVertices.Values),
        .SourceLocalTriangles = render.MeshletLocalTriangles.Get(mb.MeshletLocalTriangles.Values),
    });
}

RenderRun PublishClusterLodStorage(MeshStore &meshes, MeshStore::Record &owner, const ClusterLodBuild &build, std::span<const uint32_t> primitive_ids, RenderRun &groups) {
    auto &buffers = meshes.Render();
    groups = buffers.Insert(RenderSet::Groups, owner.GroupData, uint32_t(build.Groups.size()));
    const auto values = buffers.ClusterGroups.GetMutable({groups.Address.Value(), groups.Count});
    for (uint32_t g = 0u; g < build.Groups.size(); ++g) {
        const auto &group = build.Groups[g];
        values[g] = {.Center = group.Center, .Radius = group.Radius, .Error = group.Error};
    }
    const auto group_id = [&](uint32_t id) { return id == ClusterLodInvalid ? InvalidOffset : groups.Handle.Value + id; };
    const auto vertex_count = uint32_t(build.VertexCorners.size()), triangle_count = uint32_t(build.LocalTriangles.size() / 3u);
    const auto vertices = meshes.AppendPayload(owner, RenderPayload::VertexCorners, vertex_count);
    std::ranges::copy(build.VertexCorners, buffers.MeshletVertexCorners.GetMutable({owner.MeshletVertices.Values.Offset + vertices, vertex_count}).begin());
    const auto local_triangles = meshes.AppendPayload(owner, RenderPayload::LocalTriangles, triangle_count);
    std::ranges::copy(std::as_bytes(std::span{build.LocalTriangles}), std::as_writable_bytes(buffers.MeshletLocalTriangles.GetMutable({owner.MeshletLocalTriangles.Values.Offset + local_triangles, triangle_count})).begin());
    const auto allocation = buffers.Insert(RenderSet::Meshlets, owner.MeshletData, uint32_t(build.Clusters.size()));
    const auto records = buffers.Meshlets.GetMutable({allocation.Address.Value(), allocation.Count});
    for (uint32_t c = 0u; c < build.Clusters.size(); ++c) {
        const auto &cluster = build.Clusters[c];
        records[c] = {
            .TriangleOffset = InvalidOffset,
            .TriangleCount = cluster.TriangleCount,
            .VertexOffset = vertices + cluster.VertexOffset,
            .VertexCount = cluster.VertexCount,
            .LocalTriangleOffset = local_triangles + cluster.LocalTriangleOffset / 3u,
            .Primitive = primitive_ids[cluster.Primitive],
            .GroupIndex = group_id(cluster.GroupIndex),
            .RefinedGroup = group_id(cluster.RefinedGroup),
            .ConeAxisCutoff = cluster.ConeAxisCutoff,
            .Center = cluster.Center,
            .Radius = cluster.Radius,
        };
    }
    const auto links = buffers.GroupLinks.GetMutable({groups.Address.Value(), groups.Count});
    std::ranges::fill(links, ClusterGroupLinks{});
    for (const auto &cluster : build.Clusters) {
        if (cluster.RefinedGroup != ClusterLodInvalid) ++links[cluster.RefinedGroup].ProxyCount;
    }
    uint64_t count = build.GroupClusters.size();
    for (const auto &link : links) count += link.ProxyCount;
    if (count > UINT32_MAX) throw std::length_error("Cluster group links exceed the canonical address domain.");
    // One appended run and history capture cover every group's member/proxy run.
    const auto runs = meshes.AppendPayload(owner, RenderPayload::GroupClusters, uint32_t(count));
    const auto ids = buffers.GroupClusterIds.GetMutable({owner.GroupClusters.Values.Offset + runs, uint32_t(count)});
    for (uint32_t g = 0u, next = runs; g < links.size(); ++g) {
        auto &link = links[g];
        link.MemberOffset = next;
        next += build.Groups[g].ClusterCount;
        link.ProxyOffset = next;
        next += std::exchange(link.ProxyCount, 0u);
    }
    for (uint32_t c = 0u; c < build.Clusters.size(); ++c) {
        const auto group = build.Clusters[c].RefinedGroup;
        if (group == ClusterLodInvalid) continue;
        auto &link = links[group];
        ids[link.ProxyOffset - runs + link.ProxyCount++] = allocation.Handle.Value + c;
    }
    return allocation;
}

void CommitClusterLods(state::Scene &r, std::span<MeshStore::Record *const> owners, std::span<const ClusterLodBuild> builds) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = meshes.Render();
    // Every arena grows once for the whole batch.
    uint64_t node_count = 0u, groups = 0u, clusters = 0u, vertex_corners = 0u, local_triangles = 0u, group_ids = 0u;
    for (const auto &build : builds) {
        if (build.Groups.empty()) continue;
        node_count += build.Nodes.size();
        groups += build.Groups.size();
        clusters += build.Clusters.size();
        vertex_corners += build.VertexCorners.size();
        local_triangles += build.LocalTriangles.size();
        group_ids += build.GroupClusters.size();
        for (const auto &cluster : build.Clusters) group_ids += cluster.RefinedGroup != ClusterLodInvalid;
    }
    buffers.LodNodes.ReserveAdditional(node_count);
    buffers.LodParents.ReserveAdditional(node_count);
    buffers.ClusterGroups.ReserveAdditional(groups);
    buffers.GroupLinks.ReserveAdditional(groups);
    buffers.GroupClusterIds.ReserveAdditional(group_ids);
    buffers.Meshlets.ReserveAdditional(clusters);
    buffers.MeshletLodLeaves.ReserveAdditional(clusters);
    buffers.MeshletSpatialNodes.ReserveAdditional(clusters);
    buffers.MeshletVertexCorners.ReserveAdditional(vertex_corners);
    buffers.MeshletLocalTriangles.ReserveAdditional(local_triangles / 3u);
    for (uint32_t i = 0u; i < owners.size(); ++i) {
        auto &mb = *owners[i];
        const auto &build = builds[i];
        if (build.Groups.empty()) continue;
        assert(meshes.ClusterGroupCount(mb) == 0u);
        // Retain finest member runs and record identities.
        // Only newly constructed coarse records and traversal nodes receive new handles.
        std::vector<uint32_t> primitive_ids, retired_nodes;
        std::vector<LodNode> finest;
        meshes.ForEachPrimitive(mb, [&](ElementHandle id, const PrimitiveRecord &primitive) {
            if (primitive.Topology != 0u) return;
            primitive_ids.push_back(id.Value);
            if (primitive.LodFinestNode != InvalidOffset) retired_nodes.push_back(primitive.LodFinestNode);
            finest.push_back(primitive.LodFinestNode == InvalidOffset ? LodNode{} : buffers.LodNodes.Get({buffers.LodNodes.Address(mb.NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0]);
        });
        assert(build.PrimitiveRanges.size() == primitive_ids.size());
        std::vector<std::vector<uint32_t>> fine_ids(primitive_ids.size());
        for (uint32_t p = 0u; p < primitive_ids.size(); ++p) meshes.ForEachLodMember(mb, finest[p], [&](uint32_t id) { fine_ids[p].push_back(id); });
        // Replace only triangle leaves, retaining their member runs.
        // Line and point primitives keep their existing leaves and descriptors.
        std::ranges::sort(retired_nodes);
        buffers.LodNodes.Erase(mb.NodeData, retired_nodes);
        const auto node_run = buffers.Insert(RenderSet::Nodes, mb.NodeData, uint32_t(build.Nodes.size()));
        std::ranges::copy(build.Nodes, buffers.LodNodes.GetMutable({node_run.Address.Value(), node_run.Count}).begin());
        std::ranges::fill(buffers.LodParents.GetMutable({node_run.Address.Value(), node_run.Count}), InvalidOffset);
        RenderRun group_run;
        const auto allocation = PublishClusterLodStorage(meshes, mb, build, primitive_ids, group_run);
        const auto group_id = [&](uint32_t id) { return id == ClusterLodInvalid ? InvalidOffset : group_run.Handle.Value + id; };
        const auto group_links = buffers.GroupLinks.GetMutable({group_run.Address.Value(), group_run.Count});
        auto *cluster_ids = reinterpret_cast<uint32_t *>(buffers.GroupClusterIds.Buffer.Contents().data()) + mb.GroupClusters.Values.Offset;
        // Initial members retain canonical cluster order, and repair retains build order.
        for (uint32_t c = 0u; c < build.Clusters.size(); ++c) {
            auto &link = group_links[build.Clusters[c].GroupIndex];
            cluster_ids[link.MemberOffset + link.MemberCount++] = allocation.Handle.Value + c;
        }
        const auto meshlets = buffers.Meshlets.Entries(mb.MeshletData);
        std::vector<uint32_t> all_fine;
        for (const auto &ids : fine_ids) {
            for (const auto id : ids) all_fine.push_back(Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value());
        }
        // The finest records and their leaf entries are captured once and written in place.
        buffers.Meshlets.Buffer.CaptureWriteElements(all_fine, sizeof(MeshletRecord));
        buffers.MeshletLodLeaves.Buffer.CaptureWriteElements(all_fine, sizeof(uint32_t));
        auto *records = reinterpret_cast<MeshletRecord *>(buffers.Meshlets.Buffer.Contents().data());
        auto *fine_leaves = reinterpret_cast<uint32_t *>(buffers.MeshletLodLeaves.Buffer.Contents().data());
        auto coarse_leaves = buffers.MeshletLodLeaves.GetMutable({allocation.Address.Value(), allocation.Count});
        auto nodes = buffers.LodNodes.GetMutable({node_run.Address.Value(), node_run.Count});
        auto parents = buffers.LodParents.GetMutable({node_run.Address.Value(), node_run.Count});
        uint32_t fine_index = 0u, first_virtual = 0u;
        for (uint32_t p = 0u; p < primitive_ids.size(); ++p) {
            auto &primitive = buffers.Primitives.GetMutable({buffers.Primitives.Address(mb.PrimitiveData, ElementHandle{primitive_ids[p]}).Value(), 1u})[0];
            const auto &range = build.PrimitiveRanges[p];
            for (const auto id : fine_ids[p]) {
                const auto group = build.Level0Groups[fine_index++];
                records[Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value()].GroupIndex = group_id(group);
                auto &links = group_links[group];
                cluster_ids[links.MemberOffset + links.MemberCount++] = id;
            }
            primitive.MeshletCount = primitive.Level0Count + range.ClusterCount;
            primitive.SimplifyScale = range.SimplifyScale;
            const auto node_id = [&](uint32_t id) { return id == ClusterLodInvalid ? InvalidOffset : node_run.Handle.Value + id; };
            primitive.LodRootNode = node_id(range.RootNode);
            primitive.LodFinestNode = node_id(range.FinestNode);
            if (range.FinestNode != ClusterLodInvalid) nodes[range.FinestNode].Members = finest[p].Members;
            const auto visit = [&](auto &&self, uint32_t id) -> void {
                auto &node = nodes[id];
                if (node.ChildCount) {
                    const auto first = node.ChildOffset;
                    for (uint32_t c = 0u; c < node.ChildCount; ++c) {
                        parents[first + c] = node_run.Handle.Value + id;
                        self(self, first + c);
                    }
                    node.ChildOffset += node_run.Handle.Value;
                    return;
                }
                const auto leaf = node_run.Handle.Value + id;
                const uint32_t start = node.FirstMeshlet - first_virtual, end = start + node.MeshletCount;
                const uint32_t fine_end = std::min(end, primitive.Level0Count);
                std::vector<uint32_t> members;
                if (start < fine_end) members.assign(fine_ids[p].begin() + start, fine_ids[p].begin() + fine_end);
                for (const auto cluster : members) fine_leaves[Resolve<RenderBlockSize>(meshlets, ElementHandle{cluster}).Value()] = leaf;
                const auto coarse_start = std::max(start, primitive.Level0Count);
                for (auto c = coarse_start; c < end; ++c) {
                    const auto offset = range.FirstCluster + c - primitive.Level0Count;
                    coarse_leaves[offset] = leaf;
                    members.push_back(allocation.Handle.Value + offset);
                }
                // A leaf lists its clusters in handle order.
                std::ranges::sort(members);
                node.Members = buffers.ListMembers(mb.MemberData, members);
            };
            if (range.RootNode != ClusterLodInvalid) visit(visit, range.RootNode);
            first_virtual += primitive.MeshletCount;
        }
        for (uint32_t g = 0u; g < group_links.size(); ++g) assert(group_links[g].MemberCount == build.Groups[g].ClusterCount);
        mb.LodDepth = std::max(mb.LodDepth, build.NodeDepth);
    }
}
