#pragma once

#include "gpu/AABB.h"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshDispatchArgs.h"
#include "gpu/MeshletBlendBlockState.h"
#include "gpu/MeshletCullBlockState.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshletRoute.h"
#include "gpu/MeshletRouteState.h"
#include "gpu/MeshletWorkBlockState.h"
#include "gpu/MeshletWorkRange.h"
#include "gpu/MeshletWorkState.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/PunctualLight.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/Transform.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/VisibleMeshlet.h"
#include "gpu/WireCoverage.h"
#include "gpu/WireDrawRecord.h"
#include "gpu/WorkspaceLights.h"
#include "metal/BufferArena.h"
#include "metal/Image.h"
#include "render/MeshBuffers.h"

#include <algorithm>
#include <numeric>

struct Mesh;
struct MeshStore;


// Shared arena for per-instance GPU data (transforms, object IDs, instance states, local-space bounds).
// Uses a single RangeAllocator so all buffers share the same instance offsets.
struct InstanceArena {
    InstanceArena(mtl::BufferContext &ctx)
        : TransformBuffer(ctx, 0, SlotType::ModelBuffer),
          ObjectIdBuffer(ctx, 0, SlotType::ObjectIdBuffer),
          StateBuffer(ctx, 0, SlotType::InstanceStateBuffer),
          BoundsBuffer(ctx, 0, SlotType::Buffer),
          RecordBuffer(ctx, 0, SlotType::Buffer) {}

    Range Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;
        const uint64_t end = range.Offset + range.Count;
        EnsureCapacity(end);
        return range;
    }

    void Free(Range range) { Allocator.Free(range); }

    // Compact-erase: shift [global_index+1, range_end) down by 1 in all buffers.
    void CompactErase(uint32_t global_index, uint32_t range_end) {
        const auto count = range_end - global_index - 1;
        if (count == 0) return;
        ForEachBuffer([&](mtl::Buffer &buf, size_t sz) {
            buf.Move(uint64_t(global_index + 1) * sz, uint64_t(global_index) * sz, count * sz);
        });
    }

    // Copy `count` elements from src_offset to dst_offset across all buffers.
    void CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count) {
        if (count == 0 || src_offset == dst_offset) return;
        ForEachBuffer([&](mtl::Buffer &buf, size_t sz) {
            buf.Move(uint64_t(src_offset) * sz, uint64_t(dst_offset) * sz, count * sz);
        });
    }

    // Pre-reserve capacity for `count` additional instances to avoid per-Allocate growth.
    void ReserveAdditional(uint32_t count) { EnsureCapacity(uint64_t(Allocator.HighWaterMark()) + count); }

    void UpdateState(uint32_t index, uint8_t state) { StateBuffer.Update(as_bytes(state), uint64_t(index) * sizeof(uint8_t)); }
    const AABB &GetBounds(uint32_t index) const { return reinterpret_cast<const AABB *>(BoundsBuffer.Contents().data())[index]; }
    std::span<AABB> GetMutableBounds(Range range) const { return BoundsBuffer.GetMutableSpan<AABB>(range); }
    std::span<uint8_t> GetMutableStates() const { return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), StateBuffer.UsedSize}; }
    std::span<Transform> GetMutableTransforms() const {
        auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<Transform *>(mapped.data()), mapped.size() / sizeof(Transform)};
    }

    // Reset to empty: used size and allocator go to zero, keeping the GPU allocations for reuse.
    void Reset() {
        Allocator = {};
        ForEachBuffer([](mtl::Buffer &buf, size_t) { buf.UsedSize = 0; });
    }

    mtl::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer, BoundsBuffer, RecordBuffer;

private:
    void ForEachBuffer(auto &&fn) {
        fn(TransformBuffer, sizeof(Transform));
        fn(ObjectIdBuffer, sizeof(uint32_t));
        fn(StateBuffer, sizeof(uint8_t));
        fn(BoundsBuffer, sizeof(AABB));
        fn(RecordBuffer, sizeof(InstanceRecord));
    }

    void EnsureCapacity(uint64_t end) {
        ForEachBuffer([end](mtl::Buffer &buf, size_t sz) {
            const auto required = end * sz;
            buf.Reserve(required);
            buf.UsedSize = std::max(buf.UsedSize, required);
        });
    }

    RangeAllocator Allocator;
};

struct GpuBuffers {
    static constexpr uint32_t MaxSelectableObjects{100'000};
    // Motion blur records every step into one submission, each step reading its own view UBO
    // instance by dynamic offset. Instance 0 is the live UBO.
    static constexpr uint32_t MaxBlurSteps{64};

    // Metal requires aligned dynamic buffer offsets.
    static constexpr uint64_t ViewUboAlignment{256};
    static constexpr uint64_t ViewUboStride() {
        return (sizeof(::SceneViewUBO) + ViewUboAlignment - 1) / ViewUboAlignment * ViewUboAlignment;
    }
    static constexpr uint32_t MaxSelectableElements{16'000'000};
    static constexpr uint32_t ObjectPickBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t SelectionBitsetWords{(MaxSelectableElements + 31) / 32};

    GpuBuffers(const mtl::Context &ctx, mtl::BindlessSet &slots)
        : Ctx{ctx, slots},
          VertexBuffer{Ctx, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, SlotType::IndexBuffer},
          Meshlets{Ctx, SlotType::Buffer},
          MeshletTriangleIds{Ctx, SlotType::Buffer},
          MeshletVertexCorners{Ctx, SlotType::Buffer},
          MeshletLocalTriangles{Ctx, SlotType::Buffer},
          Primitives{Ctx, SlotType::Buffer},
          GpuInstanceSlots{Ctx, SlotType::Buffer},
          Instances{Ctx},
          MeshletWorkRanges{Ctx, 0, SlotType::Buffer},
          MeshletWorkBlocks{Ctx, 0, SlotType::Buffer},
          MeshletWorkBlockStates{Ctx, 0, SlotType::Buffer},
          MeshletWorkState{Ctx, sizeof(::MeshletWorkState), SlotType::Buffer},
          MeshletWorkDispatchArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
          VisibleMeshlets{Ctx, 0, SlotType::Buffer},
          MeshletClassifications{Ctx, 0, SlotType::Buffer},
          MeshletCullBlocks{Ctx, 0, SlotType::Buffer},
          MeshletBlendBlocks{Ctx, 0, SlotType::Buffer},
          MeshletRoutes{Ctx, sizeof(MeshletRouteState), SlotType::Buffer},
          MeshletDispatchArgs{Ctx, 0, SlotType::Buffer},
          MeshletPhase2Visible{Ctx, 0, SlotType::Buffer},
          MeshletPhase2Routes{Ctx, sizeof(MeshletRouteState), SlotType::Buffer},
          MeshletPhase2DispatchArgs{Ctx, 0, SlotType::Buffer},
          MeshletPhase2CullArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
          MeshletPhase2RangeCandidates{Ctx, 0, SlotType::Buffer},
          MeshletPhase2RangeCullArgs{Ctx, sizeof(MeshDispatchArgs), SlotType::Buffer},
          MeshletPhase2CullBlockCounts{Ctx, 0, SlotType::Buffer},
          Lights{Ctx, sizeof(PunctualLight), SlotType::LightBuffer},
          Materials{Ctx, sizeof(PBRMaterial), SlotType::MaterialBuffer},
          SceneViewUBO{Ctx, ViewUboStride() * (MaxBlurSteps + 1)},
          ViewportThemeUBO{Ctx, sizeof(ViewportTheme)},
          WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights)},
          RenderDraw{Ctx, 0, SlotType::DrawDataBuffer}, SelectionDraw{Ctx, 0, SlotType::DrawDataBuffer},
          ObjectPickKeys{Ctx, MaxSelectableObjects * sizeof(uint32_t)},
          ObjectPickSeenBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t)},
          SelectionBitset{Ctx, SelectionBitsetWords * sizeof(uint32_t)},
          MotionBlurTileIndirection{Ctx, 0},
          ElementPickKey{Ctx, sizeof(uint32_t)},
          ElementPickId{Ctx, sizeof(uint32_t)},
          WireCoverageBuffer{Ctx, 0, SlotType::Buffer},
          WireDrawRecords{Ctx, 0, SlotType::Buffer} {
    }

    void ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex) {
        FaceIndexBuffer.ReserveAdditional(face);
        EdgeIndexBuffer.ReserveAdditional(edge);
        VertexIndexBuffer.ReserveAdditional(vertex);
    }

    SlottedRange CreateIndices(std::span<const uint32_t> indices, IndexKind index_kind) {
        auto &buf = GetIndexBuffer(index_kind);
        return buf.Slotted(buf.Allocate(indices));
    }
    // Allocate an index range without writing data. Returns the range and a mutable span for direct writes.
    std::pair<SlottedRange, std::span<uint32_t>> AllocateIndices(uint32_t count, IndexKind index_kind) {
        auto &buf = GetIndexBuffer(index_kind);
        auto range = buf.Allocate(count);
        return {buf.Slotted(range), buf.GetMutable(range)};
    }
    RenderBuffers CreateRenderBuffers(std::span<const Vertex> vertices, std::span<const uint32_t> indices, IndexKind index_kind) {
        return {VertexBuffer.Allocate(vertices), CreateIndices(indices, index_kind), index_kind};
    }

    void Release(RenderBuffers &buffers) {
        VertexBuffer.Release(buffers.Vertices);
        buffers.Vertices = {};
        GetIndexBuffer(buffers.IndexType).Release(buffers.Indices);
        buffers.Indices = {};
    }

    void Release(MeshBuffers &buffers) {
        FaceIndexBuffer.Release(buffers.FaceIndices);
        buffers.FaceIndices = {};
        EdgeIndexBuffer.Release(buffers.EdgeIndices);
        buffers.EdgeIndices = {};
        VertexIndexBuffer.Release(buffers.VertexIndices);
        buffers.VertexIndices = {};
        Meshlets.Release(buffers.Meshlets);
        buffers.Meshlets = {};
        MeshletTriangleIds.Release(buffers.MeshletTriangles);
        buffers.MeshletTriangles = {};
        MeshletVertexCorners.Release(buffers.MeshletVertices);
        buffers.MeshletVertices = {};
        MeshletLocalTriangles.Release(buffers.MeshletLocalTriangles);
        buffers.MeshletLocalTriangles = {};
        Primitives.Release(buffers.Primitives);
        buffers.Primitives = {};
    }

    BufferArena<uint32_t> &GetIndexBuffer(IndexKind kind) {
        switch (kind) {
            case IndexKind::Face: return FaceIndexBuffer;
            case IndexKind::Edge: return EdgeIndexBuffer;
            case IndexKind::Vertex: return VertexIndexBuffer;
        }
    }

    // Motion blur's tile indirection table: one entry per tile, per motion direction. Sized alongside
    // the blur's other targets, so it costs nothing until a frame is blurred.
    void ResizeMotionBlurTileIndirection(mtl::Extent2D tile_extent) {
        MotionBlurTileIndirection.SetCount(2 * tile_extent.Width * tile_extent.Height);
    }

    // One counter per wire color class plus the nearest wire's depth, for every pixel.
    void ResizeWireCoverage(mtl::Extent2D extent) {
        const uint64_t words = uint64_t(extent.Width) * extent.Height * uint32_t(WireCoverage::WordsPerPixel);
        if (words != WireCoverageBuffer.Count<uint32_t>()) WireCoverageBuffer.SetCount<uint32_t>(uint32_t(words));
    }

    // Empty the scene arenas on clear so the next load's derived handles (BufferIndex, InstanceRange, morph/bone
    // ranges) rebuild from a clean baseline rather than on a prior scene's leftover residue.
    void ResetSceneArenas() {
        VertexBuffer.Reset();
        FaceIndexBuffer.Reset();
        EdgeIndexBuffer.Reset();
        VertexIndexBuffer.Reset();
        Meshlets.Reset();
        MeshletTriangleIds.Reset();
        MeshletVertexCorners.Reset();
        MeshletLocalTriangles.Reset();
        Primitives.Reset();
        GpuInstanceSlots.Reset();
        MeshletRangeCount = 0;
        MeshletInstanceCount = 0;
        MeshletTopologyMask = 0;
        // Occlusion feedback state, so the next scene's phase-1/phase-2 split replays identically
        // regardless of what rendered before the clear.
        PreviousFullCullViewProj = mat4{1};
        ArmatureDeformBuffer.Reset();
        MorphWeightBuffer.Reset();
        Instances.Reset();
    }

    mtl::BufferContext Ctx;

    // Per-scene arenas
    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    BufferArena<MeshletRecord> Meshlets;
    BufferArena<uint32_t> MeshletTriangleIds;
    BufferArena<uint32_t> MeshletVertexCorners;
    BufferArena<uint8_t> MeshletLocalTriangles;
    BufferArena<PrimitiveRecord> Primitives;
    BufferArena<uint32_t> GpuInstanceSlots;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, SlotType::MorphWeightBuffer};
    InstanceArena Instances;

    mtl::Buffer MeshletWorkRanges, MeshletWorkBlocks, MeshletWorkBlockStates, MeshletWorkState, MeshletWorkDispatchArgs;
    mtl::Buffer VisibleMeshlets, MeshletClassifications, MeshletCullBlocks, MeshletBlendBlocks, MeshletRoutes, MeshletDispatchArgs;
    mtl::Buffer MeshletPhase2Visible, MeshletPhase2Routes, MeshletPhase2DispatchArgs, MeshletPhase2CullArgs;
    mtl::Buffer MeshletPhase2RangeCandidates, MeshletPhase2RangeCullArgs, MeshletPhase2CullBlockCounts;
    uint64_t MeshletRangeCount{0};
    uint64_t MeshletInstanceCount{0};
    uint32_t MeshletTopologyMask{0};
    uint32_t MeshletDispatchChunkCount{0};

    static constexpr uint32_t MeshletDispatchChunkSize{65'535};
    static constexpr uint32_t MeshletCullBlockSize{1024};
    static constexpr uint32_t MeshletRouteCount{uint32_t(MeshletRoute::Count)};
    // One 32-lane simdgroup per phase-2 cull threadgroup, matching the shader's Phase2GroupSize.
    static constexpr uint32_t MeshletPhase2GroupSize{32};

    void EnsureMeshletVisibilityCapacity(
        uint64_t visible_count, uint64_t work_range_count, uint64_t work_meshlet_count,
        uint64_t dispatch_meshlet_count, bool sort_blend, bool two_phase
    ) {
        const auto bytes = visible_count * sizeof(VisibleMeshlet);
        VisibleMeshlets.Reserve(bytes);
        VisibleMeshlets.UsedSize = bytes;
        const auto instance_count = GpuInstanceSlots.Buffer.Count<uint32_t>();
        const auto work_block_count = (instance_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
        const auto block_count = (work_meshlet_count + MeshletCullBlockSize - 1u) / MeshletCullBlockSize;
        MeshletWorkRanges.SetCount<MeshletWorkRange>(work_range_count);
        MeshletWorkBlocks.SetCount<uint32_t>(block_count);
        MeshletWorkBlockStates.SetCount<MeshletWorkBlockState>(work_block_count);
        MeshletClassifications.SetCount<uint16_t>(work_meshlet_count);
        MeshletCullBlocks.SetCount<MeshletCullBlockState>(block_count);
        if (sort_blend) MeshletBlendBlocks.SetCount<MeshletBlendBlockState>(block_count);
        MeshletDispatchChunkCount = static_cast<uint32_t>((dispatch_meshlet_count + MeshletDispatchChunkSize - 1) / MeshletDispatchChunkSize);
        MeshletDispatchArgs.SetCount<MeshDispatchArgs>(MeshletRouteCount * MeshletDispatchChunkCount);
        if (two_phase) {
            MeshletPhase2Visible.SetCount<VisibleMeshlet>(work_meshlet_count);
            // Phase 2 conservatively uses one coverage-capable, two-sided visibility route.
            MeshletPhase2DispatchArgs.SetCount<MeshDispatchArgs>(MeshletDispatchChunkCount);
            MeshletPhase2RangeCandidates.SetCount<MeshletWorkRange>(work_range_count);
            MeshletPhase2CullBlockCounts.SetCount<uint32_t>(
                (work_meshlet_count + MeshletPhase2GroupSize - 1) / MeshletPhase2GroupSize
            );
        }
    }

    mat4 PreviousFullCullViewProj{1};

    void RebuildMeshlets(MeshBuffers &, const Mesh &, const MeshStore &);

    // Poses at the shutter's open and close, for motion blur's velocity pass. Each is a whole-buffer
    // copy of its live counterpart, so the per-draw offsets index them unchanged.
    struct VelocityPose {
        VelocityPose(mtl::BufferContext &ctx)
            : Transforms(ctx, 0, SlotType::ModelBuffer),
              ArmatureDeform(ctx, 0, SlotType::ArmatureDeformBuffer),
              MorphWeights(ctx, 0, SlotType::MorphWeightBuffer) {}

        mtl::Buffer Transforms, ArmatureDeform, MorphWeights;
        // Looking through an animated camera moves the view too, so each pose carries its own.
        mat4 ViewProj{1};
    };
    VelocityPose ShutterOpen{Ctx}, ShutterClose{Ctx};

    // Multi-step blur's pose captures: step i's shutter boundaries at [2i] and [2i+2], its centre
    // at [2i+1]. Grown on demand and kept for reuse.
    std::vector<VelocityPose> BlurPoses;
    void EnsureBlurPoses(size_t count) {
        BlurPoses.reserve(count);
        while (BlurPoses.size() < count) BlurPoses.emplace_back(Ctx);
    }

    // Point the view UBO's draw-data slot at `draw_data`, which GetDrawDataAt reads.
    void SetSceneViewDrawSlots(const mtl::Buffer &draw_data) {
        SceneViewUBO.Update(as_bytes(draw_data.Slot), offsetof(::SceneViewUBO, DrawDataSlot));
    }

    // Copy the live view UBO into ring instance `instance`.
    void SnapshotSceneViewUbo(uint32_t instance) {
        SceneViewUBO.Update(SceneViewUBO.Contents().subspan(0, sizeof(::SceneViewUBO)), ViewUboStride() * instance);
    }
    void UpdateSceneViewUboField(uint32_t instance, uint64_t field_offset, std::span<const std::byte> bytes) {
        SceneViewUBO.Update(bytes, ViewUboStride() * instance + field_offset);
    }
    uint32_t SceneViewUboOffset(uint32_t instance) const { return uint32_t(ViewUboStride() * instance); }

    // Snapshot the live pose into `dst`. Call once the scene is evaluated at the wanted time.
    void CaptureVelocityPose(VelocityPose &dst) const {
        static constexpr auto copy_whole = [](const mtl::Buffer &src, mtl::Buffer &dst) {
            dst.Reserve(src.UsedSize);
            dst.Update(src.Contents().subspan(0, src.UsedSize));
            dst.UsedSize = src.UsedSize;
        };
        copy_whole(Instances.TransformBuffer, dst.Transforms);
        copy_whole(ArmatureDeformBuffer.Buffer, dst.ArmatureDeform);
        copy_whole(MorphWeightBuffer.Buffer, dst.MorphWeights);
        dst.ViewProj = reinterpret_cast<const ::SceneViewUBO *>(SceneViewUBO.Contents().data())->ViewProj;
    }

    // Per-scene resource tables — reset via their own paths (Lights.SetCount(0) / ResetImportedTexturesAndMaterials)
    TypedBuffer<PunctualLight> Lights;
    TypedBuffer<PBRMaterial> Materials;

    // Per-frame uniforms. SceneViewUBO holds MaxBlurSteps+1 instances at SceneViewUboStride,
    // with the live state at instance 0.
    mtl::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;

    // One pass's draw data. Emissions read a draw at its own index.
    mtl::Buffer RenderDraw, SelectionDraw;

    // One entry per run of mesh instance slots sharing a deform state.
    mtl::Buffer BoundsReduceEntries{Ctx, 0, SlotType::DrawDataBuffer};
    // (entry index, tile index) per bounds threadgroup, posed entries' tiles first.
    mtl::Buffer BoundsTiles{Ctx, 0, SlotType::Buffer};
    // Per-tile partial AABBs of each entry's positions.
    mtl::Buffer BoundsPartials{Ctx, 0, SlotType::Buffer};
    // First tile index per bounds entry, locating its partials.
    mtl::Buffer BoundsEntryFirstTiles{Ctx, 0, SlotType::Buffer};
    // (entry index, tile index) per normal-derive threadgroup, the entries' face tiles in a leading prefix.
    mtl::Buffer DeriveTiles{Ctx, 0, SlotType::Buffer};
    // Current-pose vertex positions in mesh-local space, one range per posed bounds entry.
    mtl::Buffer PosedPositions{Ctx, 0, SlotType::Buffer};
    // (posed entry, global meshlet) per posed-meshlet bounds threadgroup, plus its local-space AABB output.
    mtl::Buffer PosedMeshletBoundsTiles{Ctx, 0, SlotType::Buffer};
    mtl::Buffer PosedMeshletBounds{Ctx, 0, SlotType::Buffer};
    // One entry per normal-derive dispatch item.
    // A frame holds one per posed bounds entry with triangles, and a base one-shot holds one per mesh.
    mtl::Buffer NormalDeriveEntries{Ctx, 0, SlotType::Buffer};
    // Per-instance derived normals, one range per posed derive entry.
    // The three buffers hold per-vertex smooth normals, seam-corner sector normals, and per-face fan sums.
    mtl::Buffer PosedVertexNormals{Ctx, 0, SlotType::Buffer};
    mtl::Buffer PosedSeamNormals{Ctx, 0, SlotType::Buffer};
    mtl::Buffer PosedFaceNormals{Ctx, 0, SlotType::Buffer};
    // Weight-summed authored morph normal deltas, one vec3 per posed vertex slot, present for authored-morph entries.
    mtl::Buffer PosedMorphNormalDeltas{Ctx, 0, SlotType::Buffer};
    // Instance-arena slots whose bounds draw as box wireframes, one box per listed slot.
    mtl::Buffer BoundsBoxSlots{Ctx, 0, SlotType::Buffer};

    // Group counts of the posed prelude's passes, in recorded order (their arg slot order in PreludeDispatchArgs).
    // Set on draw-list rebuild.
    struct PreludeGroups {
        static constexpr uint32_t PassCount{6};
        uint32_t PosePrepass{0}, PosedMeshletBounds{0}, DeriveFaces{0}, BoundsReduce{0}, DeriveGather{0}, BoundsCombine{0};
    };
    PreludeGroups Prelude{};
    // Holds the recorded group counts, or zeros when deform inputs are unchanged since the last submit.
    mtl::Buffer PreludeDispatchArgs{Ctx, PreludeGroups::PassCount * sizeof(MTL::DispatchThreadgroupsIndirectArguments)};
    // A deform input was written since the last submit wrote live prelude counts.
    // Deform inputs are morph weights, armature poses, transform gestures, geometry edits, and draw-list rebuilds.
    bool PreludeStale{true};
    // Scene state changed since the previous submitted Full frame. This includes visibility and
    // material changes that do not need the posed prelude but can still reveal geometry.
    bool MeshletOcclusionStale{true};

    // A visibility id carries a meshlet's position in the visible list, not a stable meshlet id, so it
    // resolves correctly only against the list the raster that wrote it saw. Every cull bumps the
    // generation, the visibility raster stamps the ids it wrote with it, and decoding compares the two.
    uint32_t MeshletVisibleGeneration{0};
    uint32_t VisibilityIdGeneration{0};

    // Selection / picking — GPU buffers + host-visible readback
    TypedBuffer<uint32_t> ObjectPickKeys, ObjectPickSeenBitset, SelectionBitset, MotionBlurTileIndirection;
    TypedBuffer<uint32_t> ElementPickKey, ElementPickId;
    mtl::Buffer WireCoverageBuffer;
    TypedBuffer<WireDrawRecord> WireDrawRecords;
};
