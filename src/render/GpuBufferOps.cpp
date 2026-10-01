#include "render/GpuBufferOps.h"

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
void GpuBuffers::ReleaseMesh(uint32_t store_id) {
    if (store_id>=Meshes.size() || !Meshes[store_id]) return;
    Release(MeshOf(store_id));
    Meshes[store_id].reset();
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

void InstanceArena::CompactErase(uint32_t global_index, uint32_t range_end) {
    CopyInstances(global_index + 1, global_index, range_end - global_index - 1);
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
    if (buffers.FaceIndices.Slot == FaceIndexBuffer.Buffer.Slot) FaceIndexBuffer.Release(buffers.FaceIndices);
    buffers.FaceIndices = {};
    ReleaseRange(EdgeIndexBuffer, buffers.EdgeIndices);
    ReleaseRange(VertexIndexBuffer, buffers.VertexIndices);
    ReleaseMeshlets(buffers);
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
    buffers.SpatialRoot=InvalidOffset;
    buffers.Level0Count = 0u;
    ActiveMeshlets.Release(buffers.PositionDirtyRoot); buffers.PositionDirtyRoot=InvalidOffset;
    ActiveMeshlets.Release(buffers.DirtyGroupRoot); buffers.DirtyGroupRoot=InvalidOffset;
    if (buffers.ElementMeshletBlockCount) {
        for (const auto block : MeshletOwnerBlocks(buffers)) ElementMeshlets[buffers.RenderTopology].Release(block);
    }
    buffers.RenderTopology = InvalidOffset;
    buffers.ElementMeshletOrigin=InvalidOffset;
    buffers.ElementMeshletBlockCount=0u;
    const auto release_group = [&](uint32_t id) {
        const auto links=GroupLinks.Get({id,1u})[0];
        GroupClusterIds.Release({links.MemberOffset,links.MemberCount});
        GroupClusterIds.Release({links.ProxyOffset,links.ProxyCount});
        ClusterGroups.Release({id,1u});
    };
    if (buffers.GroupRoot == InvalidOffset) {
        for (uint32_t i=0u; i<buffers.ClusterGroups.Count; ++i) release_group(buffers.ClusterGroups.Offset+i);
    } else ActiveMeshlets.ForEach(buffers.GroupRoot,release_group);
    ActiveMeshlets.Release(buffers.GroupRoot); buffers.GroupRoot = InvalidOffset;
    buffers.ClusterGroups = {};
    if (buffers.NodeRoot == InvalidOffset) {
        for (const auto &node : LodNodes.Get(buffers.LodNodes)) ActiveMeshlets.Release(node.MeshletRoot);
        LodNodes.Release(buffers.LodNodes);
    } else ForEachLodNode(buffers,[&](uint32_t id, const LodNode &node) {
        ActiveMeshlets.Release(node.MeshletRoot);
        LodNodes.Release({id,1u});
    });
    ActiveMeshlets.Release(buffers.NodeRoot); buffers.NodeRoot = InvalidOffset;
    buffers.LodNodes = {};
    if (buffers.MeshletRoot == InvalidOffset) {
        // Construction ranges have no published owner yet. A failed build or
        // an unadopted edit fragment still owns exactly these allocations.
        MeshletTriangleIds.Release(buffers.MeshletTriangles);
        MeshletVertexCorners.Release(buffers.MeshletVertices);
        MeshletLocalTriangles.Release(buffers.MeshletLocalTriangles);
        Meshlets.Release(buffers.Meshlets);
    } else {
        std::vector<uint32_t> handles;
        ActiveMeshlets.ForEach(buffers.MeshletRoot,[&](uint32_t handle) { handles.push_back(handle); });
        ReleaseMeshletStorage(handles);
    }
    ActiveMeshlets.Release(buffers.MeshletRoot); buffers.MeshletRoot = InvalidOffset;
    buffers.Meshlets = buffers.MeshletTriangles = buffers.MeshletVertices = buffers.MeshletLocalTriangles = {};
    buffers.CoarseVertices = buffers.CoarseLocalTriangles = {};
    // Construction owns its provisional range until membership is published
    // or an edit fragment transfers it to an existing owner.
    if (buffers.PrimitiveRoot == InvalidOffset) Primitives.Release(buffers.Primitives);
    else ActiveMeshlets.ForEach(buffers.PrimitiveRoot,[&](uint32_t id) { Primitives.Release({id,1u}); });
    ActiveMeshlets.Release(buffers.PrimitiveRoot); buffers.PrimitiveRoot = InvalidOffset;
    PrimitiveRoutes.Release(buffers.PrimitiveRoutes);
    buffers.PrimitiveRoutes = {};
    buffers.Primitives = {};
    ReleaseRange(MeshRecords, buffers.MeshRecord);
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
