#include "render/GpuBufferOps.h"
#include "Profile.h"

#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"

#include "state/Scene.h"
#include <Metal/MTLComputeCommandEncoder.hpp>

std::span<const PBRMaterial> GetMaterials(const state::Scene &r) {
    return r.Context.get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>();
}
TriangleVertexView GetFaceIndices(const state::Scene &, const Mesh &mesh) { return mesh.TriangleVertices(); }

mtl::BufferContext &GetBufferContext(state::Scene &r) { return r.Context.get<GpuBuffers>().Ctx; }

void UpdatePosedMeshletBlocks(state::Scene &r, const MeshStore::Record &owner, std::span<const uint32_t> ids) {
    const auto &index = r.Context.get<const MeshStore>().Render().ActiveMeshlets;
    std::vector<uint32_t> blocks;
    for (const auto id : ids) if (blocks.empty() || blocks.back() != id / 256u) blocks.push_back(id / 256u);
    r.Context.get<GpuBuffers>().PosedMeshletBounds.UpdateBlocks(owner.StoreId, owner.MeshletRevision, blocks, [&](uint32_t block) { return index.HasBlock(owner.MeshletRoot, block); }, owner.RenderTopology);
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
    ClearStates(range);
    return range;
}

void InstanceArena::CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count) {
    if (count == 0 || src_offset == dst_offset) return;
    ForEachBuffer([&](mtl::Buffer &buf, size_t sz) {
        buf.Move(uint64_t(src_offset) * sz, uint64_t(dst_offset) * sz, count * sz);
    });
}

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
      PreludeDispatchArgs{Ctx, PreludePassCount * sizeof(MTL::DispatchThreadgroupsIndirectArguments)},
      ObjectPickKeys{Ctx, sizeof(uint32_t)},
      ObjectPickSeenBitset{Ctx, sizeof(uint32_t)},
      ObjectBoxBitset{Ctx, sizeof(uint32_t)},
      ElementPickKey{Ctx, sizeof(uint32_t)},
      ElementPickId{Ctx, sizeof(uint32_t)} {
}

void GpuBuffers::ResetSceneArenas() {
    PosedPositions.Reset();
    PosedMorphNormalDeltas.Reset();
    PosedVertexNormals.Reset();
    PosedFaceNormals.Reset();
    PosedSectors.Reset();
    PosedMeshletBounds.Reset();
    GeometryWork.Reset();
    VertexBounds.Reset();
    GpuInstanceSlots.UsedSize = 0;
    LodNodeCount = 0;
    MeshletInstanceCount = 0;
    MeshletLodDepth = 0;
    MeshletFlagWorkByBit = {};
    FlagTallies.clear();
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

void GpuBuffers::Retally(uint32_t store_id, MeshFlagTally tally) {
    if (FlagTallies.size() <= store_id) FlagTallies.resize(store_id + 1u);
    const auto apply = [&](const MeshFlagTally &t, bool add) {
        if (!t.Meshlets) return;
        const uint64_t nodes = uint64_t(t.Nodes) * t.Instances, meshlets = uint64_t(t.Meshlets) * t.Instances;
        LodNodeCount = add ? LodNodeCount + nodes : LodNodeCount - nodes;
        MeshletInstanceCount = add ? MeshletInstanceCount + meshlets : MeshletInstanceCount - meshlets;
        for (auto bits = t.Flags & CountedMeshletFlags; bits; bits &= bits - 1u) {
            auto &work = FlagWork(bits & (~bits + 1u));
            work.Nodes = add ? work.Nodes + nodes : work.Nodes - nodes;
            work.Meshlets = add ? work.Meshlets + meshlets : work.Meshlets - meshlets;
        }
    };
    const auto previous = std::exchange(FlagTallies[store_id], tally);
    apply(previous, false);
    apply(tally, true);
    if (tally.Depth > MeshletLodDepth) MeshletLodDepth = tally.Depth;
    else if (tally.Depth < previous.Depth && previous.Depth == MeshletLodDepth) {
        MeshletLodDepth = 0u;
        for (const auto &t : FlagTallies) MeshletLodDepth = std::max(MeshletLodDepth, t.Depth);
    }
}

std::span<OverlayJob> GpuBuffers::ResizeOverlayJobs(uint32_t count) {
    const auto jobs = OverlayJobs.SetCount<OverlayJob>(count);
    OverlayJobBlocks.SetCount<uint32_t>((count + OverlayJobBlockSize - 1u) / OverlayJobBlockSize);
    VisibleOverlayJobs.SetCount<uint32_t>(count);
    *OverlayJobDispatchArgs.GetMutableSpan<MeshDispatchArgs>({0, 1}).data() = {0u, 1u, 1u};
    return jobs;
}

void GpuBuffers::EnsureMeshletVisibilityCapacity(
    MeshletCullOutput &output, uint64_t visible_count, uint64_t work_node_count, uint64_t work_meshlet_count
) {
    const auto bytes = visible_count * sizeof(VisibleMeshlet);
    output.Visible.Reserve(bytes);
    output.Visible.UsedSize = bytes;
    const auto instance_count = GpuInstanceSlots.Count<uint32_t>();
    // Classification also covers the visible entries a silhouette cull filters.
    const auto block_count = (visible_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    // A traversal level holds each drawing instance's live nodes at most once, and the final level emits at most one work range per entry.
    // The seed level's block states cover every instance slot.
    const auto frontier_count = std::max<uint64_t>(work_node_count, instance_count);
    const auto frontier_block_count = (frontier_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
    MeshletWorkRanges.SetCount<MeshletWorkRange>(work_node_count);
    MeshletWorkBlocks.SetCount<uint32_t>(block_count);
    for (auto &frontier : LodFrontiers) frontier.SetCount<LodFrontierEntry>(frontier_count);
    LodFrontierBlockStates.SetCount<LodFrontierBlockState>(frontier_block_count);
    MeshletClassifications.SetCount<uint32_t>(visible_count);
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
