#include "render/GpuBuffers.h"
#include "render/MeshletBuild.h"
#include "render/MeshletBuildGpu.h"
#include "state/Scene.h"

#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "state/Scene.h"

#include <utility>

// The mesh's shared arena locations, which ComposeDraw advances to each primitive's first triangle.
// An extras record draws its bone or joint faces with the instance transforms.
MeshRecord BuildMeshRecord(const GpuBuffers &buffers, const MeshStore &meshes, uint32_t store_id, uint32_t topology) {
    const auto &record = meshes.Get(store_id);
    const auto &arenas = meshes.Arenas();
    const auto &render = meshes.Render();
    const auto vertex_count = arenas.Vertices.Count(record.Vertices), vertex_offset = arenas.Vertices.First(record.Vertices);
    if (record.ExtrasFaces.Count) {
        return {
            .VertexSlot = arenas.Vertices.Buffer.Slot,
            .IndexSlotOffset = {render.ExtrasFaces.Buffer.Slot, record.ExtrasFaces.Offset},
            .ModelSlot = buffers.Instances.TransformBuffer.Slot,
            .VertexCountOrHeadImageSlot = vertex_count,
            .VertexOffset = vertex_offset,
        };
    }
    if (topology != 0u) {
        return {
            .VertexSlot = arenas.Vertices.Buffer.Slot,
            .IndexSlotOffset = topology == 1u ? SlotOffset{arenas.FaceCorners.Buffer.Slot, arenas.FaceCorners.First(record.FaceCorners)} : SlotOffset{},
            .ModelSlot = buffers.Instances.TransformBuffer.Slot,
            .TriangleSlot = InvalidSlot,
            .CornerColor = arenas.VertexColors.Ref(record.VertexAttributes & MeshAttributeBit_Color0),
            .Connectivity = meshes.GetConnectivityRef(store_id),
            .VertexCountOrHeadImageSlot = vertex_count,
            .VertexOffset = vertex_offset,
            .PrimitiveMaterialOffset = OffsetOrInvalid(record.PrimitiveMaterials),
            .ElementPrimitives = record.VertexPrimitivesReady ? arenas.VertexPrimitives.Ref() : ElementAttributeRef{},
        };
    }
    return {
        .VertexSlot = arenas.Vertices.Buffer.Slot,
        .IndexSlotOffset = {arenas.FaceCorners.Buffer.Slot,arenas.FaceCorners.First(record.FaceCorners)},
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .TriangleSlot = arenas.Triangles.Buffer.Slot,
        .CornerClassMode = meshes.GetCornerClassMode(store_id),
        .CustomNormals = arenas.CustomNormals.Ref(record.CornerAttributes & MeshAttributeBit_Normal),
        .CornerTangent = arenas.CornerTangents.Ref(record.CornerAttributes & MeshAttributeBit_Tangent),
        .CornerColor = arenas.CornerColors.Ref(record.CornerAttributes & MeshAttributeBit_Color0),
        .CornerUvs = {arenas.CornerUvs[0].Ref(record.CornerAttributes & MeshAttributeBit_TexCoord0), arenas.CornerUvs[1].Ref(record.CornerAttributes & MeshAttributeBit_TexCoord1), arenas.CornerUvs[2].Ref(record.CornerAttributes & MeshAttributeBit_TexCoord2), arenas.CornerUvs[3].Ref(record.CornerAttributes & MeshAttributeBit_TexCoord3)},
        .TriangleOffset = arenas.Triangles.First(record.TriangleData),
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .HalfedgeCount = arenas.FaceCorners.Count(record.FaceCorners),
        .FaceCount = arenas.FaceTriangles.Count(record.FaceData),
        .VertexCountOrHeadImageSlot = vertex_count,
        .VertexOffset = vertex_offset,
        .MorphShadingAuthored = record.MorphShadingAuthored ? 1u : 0u,
        .PrimitiveMaterialOffset = OffsetOrInvalid(record.PrimitiveMaterials),
        .ElementPrimitives = arenas.FacePrimitives.Ref(),
    };
}

void RefreshMeshBinding(state::Scene &r, uint32_t store_id) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto *record = meshes.TryGet(store_id);
    if (!record || record->RenderTopology == InvalidOffset) return;
    meshes.Render().MeshRecords.GetMutable({store_id, 1u})[0] = BuildMeshRecord(r.Context.get<const GpuBuffers>(), meshes, store_id, record->RenderTopology);
}

MeshletBuildInputs CaptureMeshletInputs(const Mesh &mesh, const MeshStore &meshes, TriangleCorners triangle_corners) {
    const uint32_t store_id = mesh.GetStoreId();

    const bool face_topology = mesh.FaceCount() > 0u;
    const auto &record = meshes.Get(store_id);
    const auto &arenas = meshes.Arenas();
    MeshletBuildInputs inputs{
        .Indices = {triangle_corners,arenas.FaceCorners.Buffer.GetSpan<uint32_t>()},
        .Vertices = arenas.Vertices.Buffer.GetSpan<Vertex>(),
        .VertexFirst = 0u,
        .DenseVertices = record.Vertices && (arenas.Vertices.Set(record.Vertices).Flags & 1u) ? arenas.Vertices.Dense(record.Vertices) : Range{},
        .Normals = meshes.GetCornerNormalView(store_id),
        .Weld = {
            .CornerClassMode = meshes.GetCornerClassMode(store_id),
            .CornerSectors = {arenas.CornerSectors.View(), triangle_corners},
            .FaceSharpness = arenas.FaceSharpness.Buffer.GetSpan<uint8_t>(),
            .TriangleFaces = {triangle_corners, arenas.HalfedgeFaces.Buffer.GetSpan<uint32_t>()},
            .CustomNormals = {arenas.CustomNormals.View(record.CornerAttributes & MeshAttributeBit_Normal), triangle_corners},
            .CornerUvs = {
                CornerAttributeView<vec2>{arenas.CornerUvs[0].View(record.CornerAttributes & (MeshAttributeBit_TexCoord0 << 0)), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[1].View(record.CornerAttributes & (MeshAttributeBit_TexCoord0 << 1)), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[2].View(record.CornerAttributes & (MeshAttributeBit_TexCoord0 << 2)), triangle_corners},
                CornerAttributeView<vec2>{arenas.CornerUvs[3].View(record.CornerAttributes & (MeshAttributeBit_TexCoord0 << 3)), triangle_corners},
            },
            .CornerTangents = CornerAttributeView<vec4>{arenas.CornerTangents.View(record.CornerAttributes & MeshAttributeBit_Tangent), triangle_corners},
            .CornerColors = CornerAttributeView<vec4>{arenas.CornerColors.View(record.CornerAttributes & MeshAttributeBit_Color0), triangle_corners},
            .MorphShadingAuthored = meshes.Get(store_id).MorphShadingAuthored,
        },
        .FaceTopology = face_topology,
    };
    return inputs;
}

ClusterLodBuild BuildMeshletClusterLod(const MeshStore &meshes, const MeshStore::Record &mb, const MeshletBuildInputs &in,
                                      std::span<const uint32_t> primitive_triangle_counts) {
    const auto &render = meshes.Render();
    std::vector<uint32_t> placed;
    meshes.ForEachPrimitive(mb,[&](uint32_t id, const PrimitiveRecord &) { placed.push_back(id); });
    if (!ClusterLodApplies(in.FaceTopology,mb.Level0Count)) return {};
    assert(meshes.ClusterGroupCount(mb) == 0u);
    std::vector<ClusterLodPrimitive> primitives;
    std::vector<ClusterLodSourceCluster> clusters;
    clusters.reserve(mb.Level0Count);
    if (!primitive_triangle_counts.empty() && primitive_triangle_counts.size()!=placed.size()) {
        throw std::invalid_argument("Live LOD primitive triangle counts do not match the owner.");
    }
    uint32_t triangle_cursor=0u;
    for (uint32_t p = 0u; p < placed.size(); ++p) {
        const auto &primitive = render.Primitives.Get({placed[p],1u})[0];
        const auto root = primitive.LodFinestNode == InvalidOffset ? InvalidOffset :
            render.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
        const uint32_t first_triangle=primitive_triangle_counts.empty() ?
            primitive.TriangleOffset-mb.MeshletTriangles.Offset : triangle_cursor;
        const uint32_t triangle_count=primitive_triangle_counts.empty() ? primitive.TriangleCount : primitive_triangle_counts[p];
        if (!primitive_triangle_counts.empty()) triangle_cursor+=triangle_count;
        primitives.push_back({
            .FirstTriangle=first_triangle,
            .TriangleCount=triangle_count,
            .FirstCluster=uint32_t(clusters.size()),.ClusterCount=render.ActiveMeshlets.Count(root),
            .Attributes=primitive.LodAttributes,
        });
        render.ActiveMeshlets.ForEach(root,[&](uint32_t id) {
            const auto &record = render.Meshlets.Get({id,1u})[0];
            clusters.push_back({
                .FirstVertex=record.VertexOffset,.VertexCount=record.VertexCount,
                .FirstLocalTriangle=record.LocalTriangleOffset,.TriangleCount=record.TriangleCount,
                .Center=record.Center,.Radius=record.Radius,.ConeCullSafe=(record.ConeAxisCutoff>>24u) != 127u,
            });
        });
    }
    if (!primitive_triangle_counts.empty() && uint64_t(triangle_cursor)*3u!=in.Indices.size()) {
        throw std::invalid_argument("Live LOD triangle handles do not cover the primitive inputs.");
    }
    return BuildClusterLod(ClusterLodMesh{
        .CornerVertices=in.Indices,.Positions=&in.Vertices.front().Position.x,.PositionStride=sizeof(Vertex),
        .VertexFirst=in.VertexFirst,.DenseVertices=in.DenseVertices,.Normals=in.Normals,.Weld=in.Weld,.Primitives=primitives,.Clusters=clusters,
        .SourceVertexCorners=render.MeshletVertexCorners.Buffer.GetSpan<uint32_t>(),
        .SourceLocalTriangles=render.MeshletLocalTriangles.Buffer.GetSpan<uint8_t>(),
    });
}

Range PublishClusterLodStorage(RenderArenas &buffers,const ClusterLodBuild &build,std::span<const uint32_t> primitive_ids,
                               Range &groups,Range &vertices,Range &local_triangles) {
    groups=buffers.ClusterGroups.Allocate(uint32_t(build.Groups.size()));
    const auto values=buffers.ClusterGroups.GetMutable(groups);
    for (uint32_t g=0u;g<build.Groups.size();++g) {
        const auto &group=build.Groups[g];
        values[g]={.Center=group.Center,.Radius=group.Radius,.Error=group.Error};
    }
    const auto group_id=[&](uint32_t id) { return id==ClusterLodInvalid ? InvalidOffset : groups.Offset+id; };
    vertices=buffers.MeshletVertexCorners.Allocate(build.VertexCorners);
    local_triangles=buffers.MeshletLocalTriangles.Allocate(build.LocalTriangles);
    const auto allocation=buffers.AllocateMeshlets(uint32_t(build.Clusters.size()));
    const auto records=buffers.Meshlets.GetMutable(allocation);
    for (uint32_t c=0u;c<build.Clusters.size();++c) {
        const auto &cluster=build.Clusters[c];
        records[c]={
            .TriangleCount=cluster.TriangleCount,.VertexOffset=vertices.Offset+cluster.VertexOffset,.VertexCount=cluster.VertexCount,
            .LocalTriangleOffset=local_triangles.Offset+cluster.LocalTriangleOffset,.Primitive=primitive_ids[cluster.Primitive],
            .GroupIndex=group_id(cluster.GroupIndex),.RefinedGroup=group_id(cluster.RefinedGroup),
            .ConeAxisCutoff=cluster.ConeAxisCutoff,.Center=cluster.Center,.Radius=cluster.Radius,
        };
    }
    buffers.GroupLinks.Mirror(groups);
    const auto links=buffers.GroupLinks.GetMutable(groups);
    std::ranges::fill(links,ClusterGroupLinks{});
    for (const auto &cluster : build.Clusters)
        if (cluster.RefinedGroup!=ClusterLodInvalid) ++links[cluster.RefinedGroup].ProxyCount;
    uint64_t count=build.GroupClusters.size();
    for (const auto &link : links) count+=link.ProxyCount;
    if (count>UINT32_MAX) throw std::length_error("Cluster group links exceed the canonical address domain.");
    // One allocation and history capture cover every group's member/proxy run.
    const auto runs=buffers.GroupClusterIds.Allocate(uint32_t(count));
    const auto ids=buffers.GroupClusterIds.GetMutable(runs);
    for (uint32_t g=0u,next=runs.Offset; g<links.size(); ++g) {
        auto &link=links[g];
        link.MemberOffset=next; next+=build.Groups[g].ClusterCount;
        link.ProxyOffset=next; next+=std::exchange(link.ProxyCount,0u);
    }
    for (uint32_t c=0u;c<build.Clusters.size();++c) {
        const auto group=build.Clusters[c].RefinedGroup;
        if (group==ClusterLodInvalid) continue;
        auto &link=links[group];
        ids[link.ProxyOffset-runs.Offset+link.ProxyCount++]=allocation.Offset+c;
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
    buffers.MeshletLocalTriangles.ReserveAdditional(local_triangles);
    // One mesh's committed records, with its primitives' retained finest roots.
    struct Commit {
        MeshStore::Record *Owner;
        const ClusterLodBuild *Build;
        std::vector<uint32_t> Finest, PrimitiveIds;
        Range Allocation;
    };
    std::vector<Commit> commits;
    std::vector<MeshletIndexEdit> ownership;
    for (uint32_t i = 0u; i < owners.size(); ++i) {
        auto &mb = *owners[i];
        const auto &build = builds[i];
        if (build.Groups.empty()) continue;
        assert(build.PrimitiveRanges.size() == meshes.PrimitiveCount(mb) && meshes.ClusterGroupCount(mb) == 0u);
        auto &commit = commits.emplace_back(Commit{.Owner = &mb, .Build = &build});
        // Retain finest roots and record identities.
        // Only newly constructed coarse records and traversal nodes receive new addresses.
        meshes.ForEachPrimitive(mb,[&](uint32_t id, const PrimitiveRecord &primitive) {
            commit.PrimitiveIds.push_back(id);
            commit.Finest.push_back(primitive.LodFinestNode == InvalidOffset ? InvalidOffset : buffers.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot);
        });
        // Without coarse groups every old node holds a finest root retained above.
        // Retire its descriptor only.
        // The replacement node takes that membership.
        std::vector<Range> released;
        meshes.ForEachLodNode(mb,[&](uint32_t id, const LodNode &) { released.push_back({id,1u}); });
        buffers.LodNodes.Release(std::move(released));
        buffers.ActiveMeshlets.Release(mb.NodeRoot); mb.NodeRoot = InvalidOffset;
        mb.LodNodes = {};
        mb.LodNodes = buffers.LodNodes.Allocate(build.Nodes);
        buffers.LodParents.Mirror(mb.LodNodes);
        std::ranges::fill(buffers.LodParents.GetMutable(mb.LodNodes),InvalidOffset);
        commit.Allocation=PublishClusterLodStorage(buffers,build,commit.PrimitiveIds,mb.ClusterGroups,mb.CoarseVertices,mb.CoarseLocalTriangles);
        ownership.push_back({.Root=mb.MeshletRoot,.Insert=commit.Allocation});
        ownership.push_back({.Insert=mb.LodNodes});
        ownership.push_back({.Insert=mb.ClusterGroups});
    }
    if (commits.empty()) return;
    buffers.ActiveMeshlets.Update(ownership);
    // Each traversal leaf takes a rank slice of its primitive's finest clusters and a run of new coarse clusters.
    std::vector<std::vector<uint32_t>> members;
    std::vector<MeshletIndexEdit> edits;
    std::vector<uint32_t> leaves;
    for (uint32_t i = 0u; i < commits.size(); ++i) {
        const auto &[owner, build_ptr, finest, primitive_ids, allocation] = commits[i];
        auto &mb = *owner;
        const auto &build = *build_ptr;
        mb.MeshletRoot = ownership[3u*i].Root; mb.NodeRoot = ownership[3u*i+1u].Root; mb.GroupRoot = ownership[3u*i+2u].Root;
        const auto group_id=[&](uint32_t id) { return id==ClusterLodInvalid ? InvalidOffset : mb.ClusterGroups.Offset+id; };
        const auto group_links=buffers.GroupLinks.GetMutable(mb.ClusterGroups);
        auto *cluster_ids=reinterpret_cast<uint32_t *>(buffers.GroupClusterIds.Buffer.Contents().data());
        // Initial members retain canonical cluster order, and repair retains build order.
        for (uint32_t c=0u;c<build.Clusters.size();++c) {
            auto &link=group_links[build.Clusters[c].GroupIndex];
            cluster_ids[link.MemberOffset+link.MemberCount++]=allocation.Offset+c;
        }
        std::vector<std::vector<uint32_t>> fine_ids(primitive_ids.size());
        std::vector<uint32_t> all_fine;
        for (uint32_t p = 0u; p < primitive_ids.size(); ++p) buffers.ActiveMeshlets.ForEach(finest[p],[&](uint32_t id) { fine_ids[p].push_back(id); });
        for (const auto &ids : fine_ids) all_fine.insert(all_fine.end(), ids.begin(), ids.end());
        // The finest records and their leaf entries are captured once and written in place.
        buffers.Meshlets.Buffer.CaptureWriteElements(all_fine, sizeof(MeshletRecord));
        buffers.MeshletLodLeaves.Buffer.CaptureWriteElements(all_fine, sizeof(uint32_t));
        auto *records = reinterpret_cast<MeshletRecord *>(buffers.Meshlets.Buffer.Contents().data());
        auto *fine_leaves = reinterpret_cast<uint32_t *>(buffers.MeshletLodLeaves.Buffer.Contents().data());
        auto coarse_leaves = buffers.MeshletLodLeaves.GetMutable(allocation);
        auto nodes = buffers.LodNodes.GetMutable(mb.LodNodes);
        auto parents = buffers.LodParents.GetMutable(mb.LodNodes);
        uint32_t fine_index = 0u, first_virtual = 0u;
        for (uint32_t p = 0u; p < primitive_ids.size(); ++p) {
            auto &primitive = buffers.Primitives.GetMutable({primitive_ids[p],1u})[0];
            const auto &range = build.PrimitiveRanges[p];
            for (const auto id : fine_ids[p]) {
                const auto group=build.Level0Groups[fine_index++];
                records[id].GroupIndex = group_id(group);
                auto &links=group_links[group];
                cluster_ids[links.MemberOffset+links.MemberCount++]=id;
            }
            primitive.MeshletCount = primitive.Level0Count+range.ClusterCount;
            primitive.SimplifyScale = range.SimplifyScale;
            const auto node_id = [&](uint32_t id) { return id == ClusterLodInvalid ? InvalidOffset : mb.LodNodes.Offset+id; };
            primitive.LodRootNode = node_id(range.RootNode); primitive.LodFinestNode = node_id(range.FinestNode);
            if (primitive.LodFinestNode != InvalidOffset) nodes[primitive.LodFinestNode-mb.LodNodes.Offset].MeshletRoot = finest[p];
            const auto visit = [&](auto &&self, uint32_t id) -> void {
                auto &node = nodes[id];
                if (node.ChildCount) {
                    const auto first = node.ChildOffset;
                    for (uint32_t c = 0u; c < node.ChildCount; ++c) {
                        parents[first+c] = mb.LodNodes.Offset+id;
                        self(self,first+c);
                    }
                    node.ChildOffset += mb.LodNodes.Offset;
                    return;
                }
                const auto leaf = mb.LodNodes.Offset+id;
                const uint32_t start = node.FirstMeshlet-first_virtual, end = start+node.MeshletCount;
                const uint32_t fine_end = std::min(end,primitive.Level0Count);
                auto &fine = members.emplace_back();
                if (start < fine_end) fine.assign(fine_ids[p].begin()+start,fine_ids[p].begin()+fine_end);
                const auto coarse_start = std::max(start,primitive.Level0Count);
                const Range coarse = coarse_start < end ? Range{allocation.Offset+range.FirstCluster+coarse_start-primitive.Level0Count,end-coarse_start} : Range{};
                for (const auto cluster : fine) fine_leaves[cluster] = leaf;
                if (coarse.Count) std::ranges::fill(coarse_leaves.subspan(coarse.Offset-allocation.Offset,coarse.Count),leaf);
                leaves.push_back(leaf);
                edits.push_back({.Insert=coarse});
            };
            if (range.RootNode != ClusterLodInvalid) visit(visit,range.RootNode);
            first_virtual += primitive.MeshletCount;
        }
        for (uint32_t g=0u; g<group_links.size(); ++g) assert(group_links[g].MemberCount==build.Groups[g].ClusterCount);
        mb.LodDepth = std::max(mb.LodDepth, build.NodeDepth);
    }
    for (uint32_t i = 0u; i < edits.size(); ++i) edits[i].Added = members[i];
    buffers.ActiveMeshlets.Update(edits);
    for (uint32_t i = 0u; i < leaves.size(); ++i) buffers.LodNodes.GetMutable({leaves[i],1u})[0].MeshletRoot = edits[i].Root;
}
