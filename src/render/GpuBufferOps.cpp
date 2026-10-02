#include "render/GpuBufferOps.h"
#include "Profile.h"

#include "mesh/Mesh.h"
#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"

#include "state/Scene.h"
#include <Metal/MTLComputeCommandEncoder.hpp>

namespace {
void ReleaseRange(auto &arena, auto &range) {
    arena.Release(range);
    range = {};
}
} // namespace

std::span<const PBRMaterial> GetMaterials(const state::Scene &r) {
    return r.Context.get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>();
}
TriangleVertexView GetFaceIndices(const state::Scene &, const Mesh &mesh) { return mesh.TriangleVertices(); }

mtl::BufferContext &GetBufferContext(state::Scene &r) { return r.Context.get<GpuBuffers>().Ctx; }
const MeshBuffers *TryMeshBuffers(const state::Scene &r, state::Entity e) {
    const auto id = DrawnStoreId(r, e);
    return id ? r.Context.get<const GpuBuffers>().TryMeshOf(*id) : nullptr;
}
const MeshBuffers &MeshBuffersOf(const state::Scene &r, state::Entity e) { return r.Context.get<const GpuBuffers>().MeshOf(*DrawnStoreId(r, e)); }
MeshBuffers &MeshBuffersOf(state::Scene &r, state::Entity e) { return r.Context.get<GpuBuffers>().MeshOf(*DrawnStoreId(r, e)); }

MeshBuffers &GpuBuffers::EmplaceMesh(uint32_t store_id, SlottedRange vertices) {
    if (MeshHistory) {
        const auto first=std::min<uint64_t>(store_id,Meshes.size());
        MeshHistory->Write(first,uint64_t(store_id)+1u-first);
    }
    if (Meshes.size() <= store_id) Meshes.resize(store_id + 1);
    assert(!Meshes[store_id]);
    return Meshes[store_id].emplace(MeshBuffers{.Vertices = vertices});
}
void GpuBuffers::ReleaseMeshes(std::span<const uint32_t> store_ids) {
    const profile::CpuScope scope{"ReleaseRenderMeshes"};
    std::vector<uint32_t> ids{store_ids.begin(), store_ids.end()};
    std::ranges::sort(ids);
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::erase_if(ids, [&](uint32_t id) { return id >= Meshes.size() || !Meshes[id]; });
    if (ids.empty()) return;
    if (MeshHistory) {
        const profile::CpuScope scope{"CaptureRenderMeshRecords"};
        ForEachIndexRun(ids, [&](size_t first, size_t count) { MeshHistory->Write(ids[first], count); });
    }
    std::vector<MeshBuffers *> records;
    for (const auto id : ids) records.push_back(&*Meshes[id]);
    Release(records);
    for (const auto id : ids) Meshes[id].reset();
}

void FreeInstanceRange(state::Scene &r, Range range) { r.Context.get<GpuBuffers>().Instances.Free(range); }

InstanceArena::InstanceArena(mtl::BufferContext &ctx)
    : TransformBuffer(ctx, 0, SlotType::ModelBuffer),
      ObjectIdBuffer(ctx, 0, SlotType::ObjectIdBuffer),
      StateBuffer(ctx, 0, SlotType::InstanceStateBuffer),
      BoundsBuffer(ctx, 0, SlotType::Buffer),
      RecordBuffer(ctx, 0, SlotType::Buffer) {}

Range InstanceArena::Allocate(uint32_t count) {
    const auto range = Allocator.Allocate(count);
    if (range.Count == 0) return range;
    EnsureCapacity(range.Offset + range.Count);
    return range;
}

void InstanceArena::CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count) {
    if (count == 0 || src_offset == dst_offset) return;
    ForEachBuffer([&](mtl::Buffer &buf, size_t sz) {
        buf.Move(uint64_t(src_offset) * sz, uint64_t(dst_offset) * sz, count * sz);
    });
}

void InstanceArena::ReserveAdditional(uint32_t count) { EnsureCapacity(uint64_t(Allocator.HighWaterMark()) + count); }

void InstanceArena::Reset() {
    Allocator = {};
    ForEachBuffer([](mtl::Buffer &buf, size_t) { buf.UsedSize = 0; });
}

void InstanceArena::EnsureCapacity(uint64_t end) {
    ForEachBuffer([end](mtl::Buffer &buf, size_t sz) {
        const auto required = end * sz;
        buf.Reserve(required);
        buf.UsedSize = std::max(buf.UsedSize, required);
    });
}

GpuBuffers::GpuBuffers(const mtl::Context &ctx, mtl::BindlessSet &slots)
    : Ctx{ctx, slots},
      VertexBuffer{Ctx, SlotType::VertexBuffer},
      FaceIndexBuffer{Ctx, SlotType::IndexBuffer},
      EdgeIndexBuffer{Ctx, SlotType::IndexBuffer},
      VertexIndexBuffer{Ctx, SlotType::IndexBuffer},
      Meshlets{Ctx, SlotType::Buffer},
      MeshletTriangleIds{Ctx, SlotType::Buffer},
      MeshletVertexCorners{Ctx, SlotType::Buffer},
      MeshletLocalTriangles{Ctx, SlotType::Buffer},
      ClusterGroups{Ctx, SlotType::Buffer},
      LodNodes{Ctx, SlotType::Buffer},
      Primitives{Ctx, SlotType::Buffer},
      MeshRecords{Ctx, SlotType::Buffer},
      GpuInstanceSlots{Ctx, 0, SlotType::Buffer},
      Instances{Ctx},
      MeshletWorkRanges{Ctx, 0, SlotType::Buffer},
      MeshletWorkBlocks{Ctx, 0, SlotType::Buffer},
      MeshletWorkState{Ctx, sizeof(::MeshletWorkState), SlotType::Buffer},
      MeshletWorkDispatchArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
      LodFrontiers{{mtl::Buffer{Ctx, 0, SlotType::Buffer}, mtl::Buffer{Ctx, 0, SlotType::Buffer}}},
      LodFrontierStates{Ctx, 2 * sizeof(::LodFrontierState), SlotType::Buffer},
      LodFrontierBlockStates{Ctx, 0, SlotType::Buffer},
      LodExpandArgs{Ctx, 2 * sizeof(MeshDispatchArgs), SlotType::Buffer},
      MeshletClassifications{Ctx, 0, SlotType::Buffer},
      MeshletCullBlocks{Ctx, 0, SlotType::Buffer},
      MeshletCoarseCount{Ctx, sizeof(uint32_t), SlotType::Buffer},
      OverlayJobs{Ctx, 0, SlotType::Buffer},
      OverlayJobBlocks{Ctx, 0, SlotType::Buffer},
      VisibleOverlayJobs{Ctx, 0, SlotType::Buffer},
      OverlayJobDispatchArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
      Lights{Ctx, sizeof(LightRecord), SlotType::LightBuffer},
      Materials{Ctx, sizeof(PBRMaterial), SlotType::MaterialBuffer},
      SceneViewUBO{Ctx, ViewUboStride() * (MaxBlurSteps + 1)},
      ViewportThemeUBO{Ctx, sizeof(ViewportTheme)},
      WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights)},
      PreludeDispatchArgs{Ctx, PreludeGroups::PassCount * sizeof(MTL::DispatchThreadgroupsIndirectArguments)},
      ObjectPickKeys{Ctx, sizeof(uint32_t)},
      ObjectPickSeenBitset{Ctx, sizeof(uint32_t)},
      ObjectBoxBitset{Ctx, sizeof(uint32_t)},
      ElementPickKey{Ctx, sizeof(uint32_t)},
      ElementPickId{Ctx, sizeof(uint32_t)} {
}

void GpuBuffers::ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex) {
    FaceIndexBuffer.ReserveAdditional(face);
    EdgeIndexBuffer.ReserveAdditional(edge);
    VertexIndexBuffer.ReserveAdditional(vertex);
}

SlottedRange GpuBuffers::CreateIndices(std::span<const uint32_t> indices, IndexKind index_kind, uint32_t vertex_first) {
    auto &buf = GetIndexBuffer(index_kind);
    const auto range = buf.Allocate(uint32_t(indices.size()));
    auto dest = buf.GetMutable(range);
    std::ranges::transform(indices, dest.begin(), [vertex_first](uint32_t v) { return vertex_first + v; });
    return buf.Slotted(range);
}

void GpuBuffers::Release(RenderBuffers &buffers) {
    ReleaseRange(VertexBuffer, buffers.Vertices);
    ReleaseRange(GetIndexBuffer(buffers.IndexType), buffers.Indices);
}

void GpuBuffers::Release(MeshBuffers &buffers) {
    auto *record = &buffers;
    Release(std::span{&record, 1u});
}
void GpuBuffers::Release(std::span<MeshBuffers *const> records) {
    std::vector<Range> faces, edges, vertices;
    for (auto *record : records) {
        if (record->FaceIndices.Slot == FaceIndexBuffer.Buffer.Slot) faces.push_back(record->FaceIndices);
        edges.push_back(record->EdgeIndices);
        vertices.push_back(record->VertexIndices);
        record->FaceIndices = record->EdgeIndices = record->VertexIndices = {};
    }
    FaceIndexBuffer.Release(std::move(faces));
    EdgeIndexBuffer.Release(std::move(edges));
    VertexIndexBuffer.Release(std::move(vertices));
    ReleaseMeshlets(records);
}

Range GpuBuffers::AllocateMeshlets(uint32_t count) {
    const auto range = Meshlets.Allocate(count);
    MeshletLodLeaves.Mirror(range);
    MeshletSpatialNodes.Mirror(range);
    return range;
}

void GpuBuffers::ReleaseMeshletStorage(std::span<const uint32_t> handles) {
    std::array<std::vector<Range>,4> ranges;
    for (const auto handle : handles) {
        const auto &record = Meshlets.Get({handle,1u})[0];
        if (record.RefinedGroup == InvalidOffset) ranges[0].push_back({record.TriangleOffset,record.TriangleCount});
        ranges[1].push_back({record.VertexOffset,record.VertexCount});
        if (record.Topology == 0u) ranges[2].push_back({record.LocalTriangleOffset,record.TriangleCount*3u});
        ranges[3].push_back({handle,1u});
    }
    MeshletTriangleIds.Release(std::move(ranges[0]));
    MeshletVertexCorners.Release(std::move(ranges[1]));
    MeshletLocalTriangles.Release(std::move(ranges[2]));
    Meshlets.Release(std::move(ranges[3]));
}

void GpuBuffers::ReservePrimitiveRoutes(MeshBuffers &mb, uint32_t count) {
    if (count <= mb.PrimitiveRoutes.Count) return;
    std::vector<uint32_t> routes(count, InvalidOffset);
    std::ranges::copy(PrimitiveRoutes.Get(mb.PrimitiveRoutes), routes.begin());
    PrimitiveRoutes.Update(mb.PrimitiveRoutes, routes);
}

std::vector<uint32_t> GpuBuffers::MeshletOwnerBlocks(const MeshBuffers &mb) const {
    std::vector<uint32_t> blocks;
    ActiveMeshlets.ForEach(mb.MeshletRoot,[&](uint32_t id) {
        const auto &record = Meshlets.Get({id,1u})[0];
        if (record.RefinedGroup != InvalidOffset || record.Topology != mb.RenderTopology) return;
        for (const auto element : MeshletTriangleIds.Get({record.TriangleOffset,record.TriangleCount}))
            if (const auto block = (mb.ElementMeshletOrigin + element) / MeshElementBlockSize; blocks.empty() || blocks.back() != block) blocks.push_back(block);
    });
    std::ranges::sort(blocks);
    blocks.erase(std::unique(blocks.begin(),blocks.end()),blocks.end());
    return blocks;
}

void GpuBuffers::ReleaseMeshlets(MeshBuffers &buffers) {
    auto *record = &buffers;
    ReleaseMeshlets(std::span{&record, 1u});
}
void GpuBuffers::ReleaseMeshlets(std::span<MeshBuffers *const> records) {
    const profile::CpuScope scope{"ReleaseRenderStorage"};
    std::vector<Range> triangles, vertices, local_triangles, meshlets, groups, group_ids, nodes, primitives, routes, meshes;
    std::vector<Range> membership_nodes, membership_leaves;
    std::array<std::vector<uint32_t>, 3> owner_blocks;
    const auto membership = ActiveMeshlets.Read();
    const auto meshlet_records = Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto lod_nodes = LodNodes.Buffer.GetSpan<LodNode>();
    const auto links = GroupLinks.Get({0u, GroupLinks.Buffer.Count<ClusterGroupLinks>()});
    const auto triangle_ids = MeshletTriangleIds.Buffer.GetSpan<uint32_t>();
    enum class Kind { Group, Node, Meshlet, Primitive };
    struct Job { Kind Type; uint32_t Owner; };
    std::vector<uint32_t> roots, additional_roots;
    std::vector<Job> jobs;
    const auto collect = [&](uint32_t root, Kind kind, uint32_t owner = 0u) {
        if (root == InvalidOffset) return;
        roots.push_back(root);
        jobs.push_back({kind, owner});
    };
    const auto release_group = [&](uint32_t id) {
        const auto &group = links[id];
        AppendRange(group_ids, {group.MemberOffset, group.MemberCount});
        AppendRange(group_ids, {group.ProxyOffset, group.ProxyCount});
        AppendRange(groups, {id, 1u});
    };
    const auto release_node = [&](uint32_t id) {
        if (lod_nodes[id].MeshletRoot != InvalidOffset) additional_roots.push_back(lod_nodes[id].MeshletRoot);
        AppendRange(nodes, {id, 1u});
    };
    {
        const profile::CpuScope scope{"CollectRenderRetirement"};
        for (uint32_t owner = 0u; owner < records.size(); ++owner) {
            const auto &b = *records[owner];
            for (const auto root : {b.PositionDirtyRoot, b.DirtyGroupRoot})
                if (root != InvalidOffset) additional_roots.push_back(root);
            if (b.GroupRoot == InvalidOffset) {
                for (uint32_t i = 0u; i < b.ClusterGroups.Count; ++i) release_group(b.ClusterGroups.Offset + i);
            } else collect(b.GroupRoot, Kind::Group);
            if (b.NodeRoot == InvalidOffset) {
                for (uint32_t i = 0u; i < b.LodNodes.Count; ++i) release_node(b.LodNodes.Offset + i);
            } else collect(b.NodeRoot, Kind::Node);
            if (b.MeshletRoot == InvalidOffset) {
                AppendRange(triangles, b.MeshletTriangles);
                AppendRange(vertices, b.MeshletVertices);
                AppendRange(local_triangles, b.MeshletLocalTriangles);
                AppendRange(meshlets, b.Meshlets);
            } else collect(b.MeshletRoot, Kind::Meshlet, owner);
            if (b.PrimitiveRoot == InvalidOffset) AppendRange(primitives, b.Primitives);
            else collect(b.PrimitiveRoot, Kind::Primitive);
            AppendRange(routes, b.PrimitiveRoutes);
            AppendRange(meshes, b.MeshRecord);
        }
        const auto release_meshlet = [&](uint32_t id, const MeshBuffers &b) {
            const auto &meshlet = meshlet_records[id];
            if (meshlet.RefinedGroup == InvalidOffset) AppendRange(triangles, {meshlet.TriangleOffset, meshlet.TriangleCount});
            AppendRange(vertices, {meshlet.VertexOffset, meshlet.VertexCount});
            if (meshlet.Topology == 0u) AppendRange(local_triangles, {meshlet.LocalTriangleOffset, meshlet.TriangleCount * 3u});
            AppendRange(meshlets, {id, 1u});
            if (b.ElementMeshletBlockCount && meshlet.RefinedGroup == InvalidOffset && meshlet.Topology == b.RenderTopology) {
                auto &blocks = owner_blocks.at(b.RenderTopology);
                for (const auto element : triangle_ids.subspan(meshlet.TriangleOffset, meshlet.TriangleCount)) {
                    const auto block = (b.ElementMeshletOrigin + element) / MeshElementBlockSize;
                    if (blocks.empty() || blocks.back() != block) blocks.push_back(block);
                }
            }
        };
        membership.CollectOwned(roots, membership_nodes, membership_leaves, [&](uint32_t root, const MeshletIndexLeaf &leaf) {
            const auto job = jobs[root];
            MeshletIndex::ForEach(leaf, [&](uint32_t handle) {
                switch (job.Type) {
                    case Kind::Group: release_group(handle); break;
                    case Kind::Node: release_node(handle); break;
                    case Kind::Meshlet: release_meshlet(handle, *records[job.Owner]); break;
                    case Kind::Primitive: AppendRange(primitives, {handle, 1u}); break;
                }
            });
        });
        membership.CollectOwned(additional_roots, membership_nodes, membership_leaves, [](uint32_t, const auto &) {});
        for (auto *record : records) {
            auto &b = *record;
            b.SpatialRoot = b.RenderTopology = b.ElementMeshletOrigin = InvalidOffset;
            b.Level0Count = b.ElementMeshletBlockCount = 0u;
            b.PositionDirtyRoot = b.DirtyGroupRoot = b.GroupRoot = b.NodeRoot = b.MeshletRoot = b.PrimitiveRoot = InvalidOffset;
            b.ClusterGroups = b.LodNodes = b.Meshlets = b.MeshletTriangles = b.MeshletVertices = b.MeshletLocalTriangles = {};
            b.CoarseVertices = b.CoarseLocalTriangles = b.PrimitiveRoutes = b.Primitives = b.MeshRecord = {};
        }
    }
    const profile::CpuScope free_scope{"FreeRenderRetirement"};
    for (uint32_t i = 0u; i < owner_blocks.size(); ++i) {
        auto &blocks = owner_blocks[i];
        std::ranges::sort(blocks);
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        ElementMeshlets[i].Release(blocks);
    }
    ActiveMeshlets.Leaves.Release(std::move(membership_leaves));
    ActiveMeshlets.Nodes.Release(std::move(membership_nodes));
    MeshletTriangleIds.Release(std::move(triangles));
    MeshletVertexCorners.Release(std::move(vertices));
    MeshletLocalTriangles.Release(std::move(local_triangles));
    Meshlets.Release(std::move(meshlets));
    GroupClusterIds.Release(std::move(group_ids));
    ClusterGroups.Release(std::move(groups));
    LodNodes.Release(std::move(nodes));
    Primitives.Release(std::move(primitives));
    PrimitiveRoutes.Release(std::move(routes));
    MeshRecords.Release(std::move(meshes));
}

void GpuBuffers::ResetSceneArenas() {
    PosedPositions.Reset();
    PosedMorphNormalDeltas.Reset();
    PosedVertexNormals.Reset();
    PosedFaceNormals.Reset();
    PosedSectors.Reset();
    PosedMeshletBounds.Reset();
    VertexBuffer.Reset();
    FaceIndexBuffer.Reset();
    EdgeIndexBuffer.Reset();
    VertexIndexBuffer.Reset();
    Meshlets.Reset();
    MeshletSpatialNodes.Reset();
    ActiveMeshlets.Reset();
    MeshletTriangleIds.Reset();
    GeometryWork.Reset();
    for (auto &owners : ElementMeshlets) { owners.Values.Reset(); owners.Blocks.Reset(); owners.Owners.Reset(); }
    VertexBounds.Reset();
    MeshletVertexCorners.Reset();
    MeshletLocalTriangles.Reset();
    ClusterGroups.Reset();
    LodNodes.Reset();
    MeshletLodLeaves.Reset(); LodParents.Reset();
    GroupLinks.Reset(); GroupClusterIds.Reset();
    Primitives.Reset();
    PrimitiveRoutes.Reset();
    MeshRecords.Reset();
    if (MeshHistory) MeshHistory->Write(0u,Meshes.size());
    Meshes.clear();
    GpuInstanceSlots.UsedSize = 0;
    LodNodeCount = 0;
    MeshletInstanceCount = 0;
    if (MeshletLodDepth && LodDepthHistory) LodDepthHistory->Write(0u,1u);
    MeshletLodDepth = 0;
    MeshletFlagWorkByBit = {};
    MeshletTopologyMask = 0;
    OverlayJobs.UsedSize = 0;
    OverlayJobBlocks.UsedSize = 0;
    VisibleOverlayJobs.UsedSize = 0;
    // Reset occlusion feedback for deterministic two-phase culling after a scene clear.
    PreviousFullCullViewProj = mat4{1};
    ArmatureDeformBuffer.Reset();
    MorphWeightBuffer.Reset();
    Instances.Reset();
}

void GpuBuffers::SetOverlayJobs(std::span<const OverlayJob> jobs) {
    OverlayJobs.SetCount<OverlayJob>(uint32_t(jobs.size()));
    if (!jobs.empty()) OverlayJobs.Update(as_bytes(jobs));
    OverlayJobBlocks.SetCount<uint32_t>(
        uint32_t((jobs.size() + OverlayJobBlockSize - 1u) / OverlayJobBlockSize)
    );
    VisibleOverlayJobs.SetCount<uint32_t>(uint32_t(jobs.size()));
    *OverlayJobDispatchArgs.GetMutableSpan<MeshDispatchArgs>({0, 1}).data() = {0u, 1u, 1u};
}

void GpuBuffers::EnsureMeshletVisibilityCapacity(
    MeshletCullOutput &output, uint64_t visible_count, uint64_t work_node_count, uint64_t work_meshlet_count
) {
    const auto bytes = visible_count * sizeof(VisibleMeshlet);
    output.Visible.Reserve(bytes);
    output.Visible.UsedSize = bytes;
    const auto instance_count = GpuInstanceSlots.Count<uint32_t>();
    const auto block_count = (work_meshlet_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    // A traversal level holds each drawing instance's live nodes at most once, and the final level emits at most one work range per entry.
    // The seed level's block states cover every instance slot.
    const auto frontier_count = std::max<uint64_t>(work_node_count, instance_count);
    const auto frontier_block_count = (frontier_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    MeshletWorkRanges.SetCount<MeshletWorkRange>(work_node_count);
    MeshletWorkBlocks.SetCount<uint32_t>(block_count);
    for (auto &frontier : LodFrontiers) frontier.SetCount<LodFrontierEntry>(frontier_count);
    LodFrontierBlockStates.SetCount<LodFrontierBlockState>(frontier_block_count);
    MeshletClassifications.SetCount<uint32_t>(work_meshlet_count);
    MeshletCullBlocks.SetCount<MeshletCullBlockState>(block_count);
    output.ChunkCount = static_cast<uint32_t>((work_meshlet_count + MeshletDispatchChunkSize - 1) / MeshletDispatchChunkSize);
    output.DispatchArgs.SetCount<MeshDispatchArgs>(MeshletRouteCount * output.ChunkCount);
}

void GpuBuffers::CaptureRenderPose(RenderPose &dst) const {
    static constexpr auto copy_whole = [](const mtl::Buffer &src, mtl::Buffer &dst) {
        dst.Reserve(src.UsedSize);
        dst.Update(src.Contents().subspan(0, src.UsedSize));
        dst.UsedSize = src.UsedSize;
    };
    copy_whole(Instances.TransformBuffer, dst.Transforms);
    copy_whole(ArmatureDeformBuffer.Buffer, dst.ArmatureDeform);
    copy_whole(MorphWeightBuffer.Buffer, dst.MorphWeights);
    copy_whole(Lights, dst.Lights);
}
