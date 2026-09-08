#pragma once

#include "gpu/AABB.h"
#include "gpu/ClusterGroup.h"
#include "gpu/InstanceRecord.h"
#include "gpu/LodFrontierBlockState.h"
#include "gpu/LodFrontierEntry.h"
#include "gpu/LodFrontierState.h"
#include "gpu/LodNode.h"
#include "gpu/MeshDispatchArgs.h"
#include "gpu/MeshletCullBlockState.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshletRoute.h"
#include "gpu/MeshletRouteState.h"
#include "gpu/MeshletWorkRange.h"
#include "gpu/MeshletWorkState.h"
#include "gpu/OverlayJob.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/PunctualLight.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/Transform.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/VisibleMeshlet.h"
#include "gpu/WorkspaceLights.h"
#include "metal/BufferArena.h"
#include "render/ClusterLod.h"
#include "render/MeshBuffers.h"
#include "viewport/RenderView.h"

#include <algorithm>
#include <array>
#include <bit>

// Per-instance GPU data behind one RangeAllocator, so every buffer shares the same instance offsets.
struct InstanceArena {
    InstanceArena(mtl::BufferContext &ctx);

    Range Allocate(uint32_t count);
    void Free(Range range) { Allocator.Free(range); }

    void CompactErase(uint32_t global_index, uint32_t range_end);
    void CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count);
    void ReserveAdditional(uint32_t count);
    void UpdateState(uint32_t index, uint8_t state) { StateBuffer.Update(as_bytes(state), uint64_t(index) * sizeof(uint8_t)); }
    const AABB &GetBounds(uint32_t index) const { return reinterpret_cast<const AABB *>(BoundsBuffer.Contents().data())[index]; }
    std::span<AABB> GetMutableBounds(Range range) const { return BoundsBuffer.GetMutableSpan<AABB>(range); }
    std::span<uint8_t> GetMutableStates() const { return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), StateBuffer.UsedSize}; }
    std::span<Transform> GetMutableTransforms() const {
        auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<Transform *>(mapped.data()), mapped.size() / sizeof(Transform)};
    }

    // Zero the used sizes and the allocator, keeping the GPU allocations for reuse.
    void Reset();

    mtl::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer, BoundsBuffer, RecordBuffer;

private:
    void ForEachBuffer(auto &&fn) {
        fn(TransformBuffer, sizeof(Transform));
        fn(ObjectIdBuffer, sizeof(uint32_t));
        fn(StateBuffer, sizeof(uint8_t));
        fn(BoundsBuffer, sizeof(AABB));
        fn(RecordBuffer, sizeof(InstanceRecord));
    }

    void EnsureCapacity(uint64_t end);

    RangeAllocator Allocator;
};

struct GpuBuffers {
    static constexpr uint32_t MaxSelectableObjects{100'000};
    // Motion-blur steps use separate dynamic view-UBO offsets in one submission.
    // Instance zero remains active.
    static constexpr uint32_t MaxBlurSteps{64};

    // Metal requires aligned dynamic buffer offsets.
    static constexpr uint64_t ViewUboAlignment{256};
    static constexpr uint64_t ViewUboStride() {
        return (sizeof(::SceneViewUBO) + ViewUboAlignment - 1) / ViewUboAlignment * ViewUboAlignment;
    }
    static constexpr uint32_t ObjectPickBitsetWords{(MaxSelectableObjects + 31) / 32};

    GpuBuffers(const mtl::Context &ctx, mtl::BindlessSet &slots);

    void ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex);

    SlottedRange CreateIndices(std::span<const uint32_t> indices, IndexKind index_kind);
    std::pair<SlottedRange, std::span<uint32_t>> AllocateIndices(uint32_t count, IndexKind index_kind);
    RenderBuffers CreateRenderBuffers(std::span<const Vertex> vertices, std::span<const uint32_t> indices, IndexKind index_kind);

    void Release(RenderBuffers &buffers);
    void Release(MeshBuffers &buffers);
    void ReleaseMeshlets(MeshBuffers &buffers);

    BufferArena<uint32_t> &GetIndexBuffer(IndexKind kind) {
        switch (kind) {
            case IndexKind::Face: return FaceIndexBuffer;
            case IndexKind::Edge: return EdgeIndexBuffer;
            case IndexKind::Vertex: return VertexIndexBuffer;
        }
    }

    // Reset derived handles to a deterministic scene-load baseline.
    void ResetSceneArenas();

    mtl::BufferContext Ctx;

    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    BufferArena<MeshletRecord> Meshlets;
    BufferArena<uint32_t> MeshletTriangleIds;
    BufferArena<uint32_t> MeshletVertexCorners;
    BufferArena<uint8_t> MeshletLocalTriangles;
    BufferArena<uint32_t> MeshletEditEdgeIds;
    // The cluster LOD DAG: one group per simplification step, and the selection forest over them.
    BufferArena<ClusterGroup> ClusterGroups;
    BufferArena<LodNode> LodNodes;
    BufferArena<PrimitiveRecord> Primitives;
    mtl::Buffer GpuInstanceSlots;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, SlotType::MorphWeightBuffer};
    InstanceArena Instances;

    mtl::Buffer MeshletWorkRanges, MeshletWorkBlocks, MeshletWorkState, MeshletWorkDispatchArgs;
    // Span-tree traversal alternates frontiers and stores each level's size, block prefixes, and indirect arguments.
    std::array<mtl::Buffer, 2> LodFrontiers;
    mtl::Buffer LodFrontierStates, LodFrontierBlockStates, LodExpandArgs;
    mtl::Buffer VisibleMeshlets, MeshletClassifications, MeshletCullBlocks, MeshletRoutes, MeshletDispatchArgs;
    // Coarse clusters the last cull's cut selected, which the classification accumulates.
    mtl::Buffer MeshletCoarseCount;
    // Persistent procedural line jobs, deterministically compacted into one indirect submission.
    mtl::Buffer OverlayJobs, OverlayJobBlocks, VisibleOverlayJobs, OverlayJobDispatchArgs;
    uint64_t MeshletRangeCount{0};
    uint64_t MeshletInstanceCount{0};
    // Maximum traversal depth among resident mesh span trees.
    uint32_t MeshletLodDepth{0};
    uint32_t MeshletTopologyMask{0};
    uint32_t MeshletDispatchChunkCount{0};

    // Maintained totals for culls restricted to one instance flag.
    struct MeshletFlagWork {
        uint64_t Ranges{0}, Meshlets{0};
    };
    // One entry per MeshletInstanceFlag bit, indexed by that bit's position.
    static constexpr size_t MeshletInstanceFlagCount = std::bit_width(uint32_t(MeshletInstanceFlag::SoundPoint));
    std::array<MeshletFlagWork, MeshletInstanceFlagCount> MeshletFlagWorkByBit{};

    MeshletFlagWork &FlagWork(uint32_t flag) { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }
    const MeshletFlagWork &FlagWork(uint32_t flag) const { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }

    static constexpr uint32_t MeshletDispatchChunkSize{65'535};
    static constexpr uint32_t MeshletCullBlockSize{1024};
    static constexpr uint32_t MeshletRouteCount{uint32_t(MeshletRoute::Count)};
    static constexpr uint32_t OverlayJobBlockSize{256};

    void SetOverlayJobs(std::span<const OverlayJob> jobs);

    void EnsureMeshletVisibilityCapacity(
        uint64_t visible_count, uint64_t work_range_count, uint64_t work_meshlet_count
    );

    mat4 PreviousFullCullViewProj{1};

    // Each shutter sample retains its evaluated geometry in UMA buffers.
    struct RenderPose {
        RenderPose(mtl::BufferContext &ctx)
            : Transforms(ctx, 0, SlotType::ModelBuffer),
              ArmatureDeform(ctx, 0, SlotType::ArmatureDeformBuffer),
              MorphWeights(ctx, 0, SlotType::MorphWeightBuffer),
              Lights(ctx, 0, SlotType::LightBuffer) {}

        mtl::Buffer Transforms, ArmatureDeform, MorphWeights, Lights;

        void ApplyTo(::SceneViewUBO &view) const {
            view.ModelSlotOverride = Transforms.Slot;
            view.ArmatureDeformSlot = ArmatureDeform.Slot;
            view.MorphWeightsSlot = MorphWeights.Slot;
            view.LightSlot = Lights.Slot;
        }
    };

    // One pose per shutter sample.
    std::vector<RenderPose> BlurPoses;
    uint32_t SceneViewUboOffset(uint32_t instance) const { return uint32_t(ViewUboStride() * instance); }

    // Requires the scene evaluated at the capture time.
    void CaptureRenderPose(RenderPose &dst) const;

    // Per-scene resource tables, reset through their own paths rather than ResetSceneArenas.
    TypedBuffer<PunctualLight> Lights;
    TypedBuffer<PBRMaterial> Materials;

    RenderView FrameView{};
    // SceneViewUBO stores the live state at instance zero and one aligned instance per blur step.
    mtl::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;

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
    // Contains one entry per posed triangle range or one per mesh during base derivation.
    mtl::Buffer NormalDeriveEntries{Ctx, 0, SlotType::Buffer};
    // Per-instance derived normals, one range per posed derive entry.
    // Stores smooth vertex normals, corner-sector normals, and face fan sums in separate buffers.
    mtl::Buffer PosedVertexNormals{Ctx, 0, SlotType::Buffer};
    mtl::Buffer PosedSeamNormals{Ctx, 0, SlotType::Buffer};
    mtl::Buffer PosedFaceNormals{Ctx, 0, SlotType::Buffer};
    // Weight-summed authored morph normal deltas, one vec3 per posed vertex slot, present for authored-morph entries.
    mtl::Buffer PosedMorphNormalDeltas{Ctx, 0, SlotType::Buffer};
    // Group counts of the posed prelude's passes, in recorded order (their arg slot order in PreludeDispatchArgs).
    // Set when persistent scene descriptors refresh.
    struct PreludeGroups {
        static constexpr uint32_t PassCount{6};
        uint32_t PosePrepass{0}, PosedMeshletBounds{0}, DeriveFaces{0}, BoundsReduce{0}, DeriveGather{0}, BoundsCombine{0};

        // Gather and combine reuse the preceding stage's dispatch count.
        bool HasWork() const { return PosePrepass > 0 || PosedMeshletBounds > 0 || DeriveFaces > 0 || BoundsReduce > 0; }
    };
    PreludeGroups Prelude{};
    // Stores recorded group counts or zeros for unchanged deform inputs.
    mtl::Buffer PreludeDispatchArgs;
    // A deform input was written since the last submit wrote live prelude counts.
    // Deform inputs are morph weights, armature poses, transform gestures, geometry edits, and scene refreshes.
    bool PreludeStale{true};
    // Tracks visibility or material changes that can reveal geometry without requiring the posed prelude.
    bool MeshletOcclusionStale{true};
    // Tracks whether edge or vertex indices require lazy construction for overlay rendering.
    bool DrewElementIndices{false};

    // Visibility IDs index the visible list and require matching cull and raster generations for decoding.
    uint32_t MeshletVisibleGeneration{0};
    struct VisibilityState {
        uint32_t Generation{InvalidOffset};
        bool ExcludesTransmission{false};
        bool operator==(const VisibilityState &) const = default;
    } Visibility;

    TypedBuffer<uint32_t> ObjectPickKeys, ObjectPickSeenBitset, ObjectBoxBitset;
    uint32_t ObjectPickEpochTag{}; // Zero clears the persistent keys before the first pick and after wraparound.
    TypedBuffer<uint32_t> ElementPickKey, ElementPickId;
    BufferArena<uint32_t> GeometryWork{Ctx, SlotType::Buffer};
    mtl::Buffer GeometryNormalEntries{Ctx, 0, SlotType::Buffer};
    BufferArena<uint32_t> ElementMeshlets{Ctx, SlotType::Buffer};
    BufferArena<AABB> BoundsParents{Ctx, SlotType::Buffer};
    mtl::Buffer EditSelectionPositionSums;
    mtl::Buffer WireCoverageBuffer{Ctx, 0, SlotType::Buffer};
};
