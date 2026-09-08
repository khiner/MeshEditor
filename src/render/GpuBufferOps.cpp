#include "render/GpuBufferOps.h"

#include "mesh/Mesh.h"
#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"

#include <Metal/MTLComputeCommandEncoder.hpp>
#include <entt/entity/registry.hpp>

namespace {
void ReleaseRange(auto &arena, auto &range) {
    arena.Release(range);
    range = {};
}
} // namespace

std::span<PBRMaterial> GetMaterials(entt::registry &r) {
    auto &materials = r.ctx().get<GpuBuffers>().Materials;
    return {materials.Data(), materials.Count()};
}
std::span<const uint32_t> GetFaceIndices(const entt::registry &r, const Mesh &mesh, const MeshBuffers &buffers) {
    const auto corners = mesh.CornerVertices();
    if (corners.size() == mesh.TriangleIndexCount()) return corners;
    return r.ctx().get<const GpuBuffers>().FaceIndexBuffer.Get(buffers.FaceIndices);
}
std::span<const PunctualLight> GetLights(entt::registry &r) {
    const auto &lights = r.ctx().get<GpuBuffers>().Lights;
    return {lights.Data(), lights.Count()};
}
PunctualLight GetLight(entt::registry &r, uint32_t index) { return r.ctx().get<GpuBuffers>().Lights.Get(index); }
mtl::BufferContext &GetBufferContext(entt::registry &r) { return r.ctx().get<GpuBuffers>().Ctx; }

void ReleaseMeshBuffers(entt::registry &r, MeshBuffers &mb) { r.ctx().get<GpuBuffers>().Release(mb); }

void FreeInstanceRange(entt::registry &r, Range range) { r.ctx().get<GpuBuffers>().Instances.Free(range); }
void ReleaseEdgeIndices(entt::registry &r, const SlottedRange &indices) { r.ctx().get<GpuBuffers>().EdgeIndexBuffer.Release(indices); }

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
      MeshletEditEdgeIds{Ctx, SlotType::Buffer},
      ClusterGroups{Ctx, SlotType::Buffer},
      LodNodes{Ctx, SlotType::Buffer},
      Primitives{Ctx, SlotType::Buffer},
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
      VisibleMeshlets{Ctx, 0, SlotType::Buffer},
      MeshletClassifications{Ctx, 0, SlotType::Buffer},
      MeshletCullBlocks{Ctx, 0, SlotType::Buffer},
      MeshletRoutes{Ctx, sizeof(MeshletRouteState), SlotType::Buffer},
      MeshletDispatchArgs{Ctx, 0, SlotType::Buffer},
      MeshletCoarseCount{Ctx, sizeof(uint32_t), SlotType::Buffer},
      OverlayJobs{Ctx, 0, SlotType::Buffer},
      OverlayJobBlocks{Ctx, 0, SlotType::Buffer},
      VisibleOverlayJobs{Ctx, 0, SlotType::Buffer},
      OverlayJobDispatchArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
      Lights{Ctx, sizeof(PunctualLight), SlotType::LightBuffer},
      Materials{Ctx, sizeof(PBRMaterial), SlotType::MaterialBuffer},
      SceneViewUBO{Ctx, ViewUboStride() * (MaxBlurSteps + 1)},
      ViewportThemeUBO{Ctx, sizeof(ViewportTheme)},
      WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights)},
      PreludeDispatchArgs{Ctx, PreludeGroups::PassCount * sizeof(MTL::DispatchThreadgroupsIndirectArguments)},
      ObjectPickKeys{Ctx, MaxSelectableObjects * sizeof(uint32_t)},
      ObjectPickSeenBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t)},
      ObjectBoxBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t)},
      ElementPickKey{Ctx, sizeof(uint32_t)},
      ElementPickId{Ctx, sizeof(uint32_t)},
      EditSelectionPositionSums{Ctx, 0, SlotType::Buffer} {
}

void GpuBuffers::ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex) {
    FaceIndexBuffer.ReserveAdditional(face);
    EdgeIndexBuffer.ReserveAdditional(edge);
    VertexIndexBuffer.ReserveAdditional(vertex);
}

SlottedRange GpuBuffers::CreateIndices(std::span<const uint32_t> indices, IndexKind index_kind) {
    auto &buf = GetIndexBuffer(index_kind);
    return buf.Slotted(buf.Allocate(indices));
}

std::pair<SlottedRange, std::span<uint32_t>> GpuBuffers::AllocateIndices(uint32_t count, IndexKind index_kind) {
    auto &buf = GetIndexBuffer(index_kind);
    auto range = buf.Allocate(count);
    return {buf.Slotted(range), buf.GetMutable(range)};
}

RenderBuffers GpuBuffers::CreateRenderBuffers(std::span<const Vertex> vertices, std::span<const uint32_t> indices, IndexKind index_kind) {
    return {VertexBuffer.Allocate(vertices), CreateIndices(indices, index_kind), index_kind};
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

void GpuBuffers::ReleaseMeshlets(MeshBuffers &buffers) {
    ReleaseRange(ClusterGroups, buffers.ClusterGroups);
    ReleaseRange(LodNodes, buffers.LodNodes);
    ReleaseRange(MeshletVertexCorners, buffers.CoarseVertices);
    ReleaseRange(MeshletLocalTriangles, buffers.CoarseLocalTriangles);
    ReleaseRange(Meshlets, buffers.Meshlets);
    ReleaseRange(MeshletTriangleIds, buffers.MeshletTriangles);
    ReleaseRange(MeshletVertexCorners, buffers.MeshletVertices);
    ReleaseRange(MeshletLocalTriangles, buffers.MeshletLocalTriangles);
    ReleaseRange(MeshletEditEdgeIds, buffers.MeshletEditEdges);
    ReleaseRange(Primitives, buffers.Primitives);
}

void GpuBuffers::ResetSceneArenas() {
    VertexBuffer.Reset();
    FaceIndexBuffer.Reset();
    EdgeIndexBuffer.Reset();
    VertexIndexBuffer.Reset();
    Meshlets.Reset();
    MeshletTriangleIds.Reset();
    GeometryWork.Reset();
    ElementMeshlets.Reset();
    BoundsParents.Reset();
    MeshletVertexCorners.Reset();
    MeshletLocalTriangles.Reset();
    MeshletEditEdgeIds.Reset();
    ClusterGroups.Reset();
    LodNodes.Reset();
    Primitives.Reset();
    GpuInstanceSlots.UsedSize = 0;
    MeshletRangeCount = 0;
    MeshletInstanceCount = 0;
    MeshletLodDepth = 0;
    MeshletFlagWorkByBit = {};
    MeshletTopologyMask = 0;
    OverlayJobs.UsedSize = 0;
    OverlayJobBlocks.UsedSize = 0;
    VisibleOverlayJobs.UsedSize = 0;
    DrewElementIndices = false;
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
    uint64_t visible_count, uint64_t work_range_count, uint64_t work_meshlet_count
) {
    const auto bytes = visible_count * sizeof(VisibleMeshlet);
    VisibleMeshlets.Reserve(bytes);
    VisibleMeshlets.UsedSize = bytes;
    const auto instance_count = GpuInstanceSlots.Count<uint32_t>();
    const auto block_count = (work_meshlet_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    // Two entries per leaf cover interior levels.
    // Per-range padding covers partial leaves and paths to the root.
    const auto node_count = 2u * (work_meshlet_count / ClusterLodSpanLeafRecords) + 8u * work_range_count + 64u;
    const auto frontier_count = std::max<uint64_t>(node_count, instance_count);
    const auto frontier_block_count = (frontier_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    MeshletWorkRanges.SetCount<MeshletWorkRange>(node_count);
    MeshletWorkBlocks.SetCount<uint32_t>(block_count);
    for (auto &frontier : LodFrontiers) frontier.SetCount<LodFrontierEntry>(frontier_count);
    LodFrontierBlockStates.SetCount<LodFrontierBlockState>(frontier_block_count);
    MeshletClassifications.SetCount<uint32_t>(work_meshlet_count);
    MeshletCullBlocks.SetCount<MeshletCullBlockState>(block_count);
    MeshletDispatchChunkCount = static_cast<uint32_t>((work_meshlet_count + MeshletDispatchChunkSize - 1) / MeshletDispatchChunkSize);
    MeshletDispatchArgs.SetCount<MeshDispatchArgs>(MeshletRouteCount * MeshletDispatchChunkCount);
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
