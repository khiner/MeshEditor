#pragma once

#include "Range.h"
#include "RangeAllocator.h"
#include "SlottedRange.h"
#include "gpu/AABB.h"
#include "gpu/BindlessBindings.h"
#include "gpu/ClusterGroup.h"
#include "gpu/ClusterGroupLinks.h"
#include "gpu/InstanceRecord.h"
#include "gpu/LightRecord.h"
#include "gpu/LodFrontierBlockState.h"
#include "gpu/LodFrontierEntry.h"
#include "gpu/LodFrontierState.h"
#include "gpu/LodNode.h"
#include "gpu/MeshDispatchArgs.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshletCullBlockState.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshletRoute.h"
#include "gpu/MeshletRouteState.h"
#include "gpu/MeshletSpatialNode.h"
#include "gpu/MeshletWorkRange.h"
#include "gpu/MeshletWorkState.h"
#include "gpu/OverlayJob.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PrimitiveRecord.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/Transform.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/VisibleMeshlet.h"
#include "gpu/WorkspaceLights.h"
#include "mesh/ElementAttribute.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"
#include "render/ClusterLod.h"
#include "render/MeshBuffers.h"
#include "render/MeshletIndex.h"
#include "render/PoseAttributeStore.h"
#include "render/VertexBoundsStore.h"
#include "viewport/RenderView.h"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>

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

// A cull owns only the list and indirect arguments consumed by later draws.
// Frontier traversal and classification scratch are shared between sequential culls.
struct MeshletCullOutput {
    explicit MeshletCullOutput(mtl::BufferContext &ctx)
        : Visible{ctx, 0, SlotType::Buffer},
          Routes{ctx, sizeof(MeshletRouteState), SlotType::Buffer},
          DispatchArgs{ctx, 0, SlotType::Buffer} {}

    mtl::Buffer Visible, Routes, DispatchArgs;
    uint32_t ChunkCount{};
};

struct GpuBuffers {
    static constexpr uint32_t MaxSelectableObjects{1u << 20};
    // Motion-blur steps use separate dynamic view-UBO offsets in one submission.
    // Instance zero remains active.
    static constexpr uint32_t MaxBlurSteps{64};

    // Metal requires aligned dynamic buffer offsets.
    static constexpr uint64_t ViewUboAlignment{256};
    static constexpr uint64_t ViewUboStride() {
        return (sizeof(::SceneViewUBO) + ViewUboAlignment - 1) / ViewUboAlignment * ViewUboAlignment;
    }

    GpuBuffers(const mtl::Context &ctx, mtl::BindlessSet &slots);
    void Track(store::History &);
    // Rebinds the render records a restore changed and returns their store IDs.
    std::vector<uint32_t> RestoreMeshBindings(state::Scene &);
    void RefreshMeshBinding(state::Scene &,uint32_t store_id);

    void ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex);

    SlottedRange CreateIndices(std::span<const uint32_t> indices, IndexKind index_kind, uint32_t vertex_first);

    void Release(RenderBuffers &buffers);
    void Release(MeshBuffers &buffers);
    void ReleaseMeshlets(MeshBuffers &buffers);
    // Allocates `count` cluster records and extends the LOD leaf and spatial mirrors over them.
    Range AllocateMeshlets(uint32_t count);
    // Releases the clusters' payload ranges, which their host records name, and their identities.
    void ReleaseMeshletStorage(std::span<const uint32_t> handles);
    uint32_t MeshletCount(const MeshBuffers &mb) const { return ActiveMeshlets.Count(mb.MeshletRoot); }
    uint32_t FirstMeshlet(const MeshBuffers &mb) const { return ActiveMeshlets.First(mb.MeshletRoot); }
    uint32_t PrimitiveCount(const MeshBuffers &mb) const { return ActiveMeshlets.Count(mb.PrimitiveRoot); }
    void ForEachPrimitive(const MeshBuffers &mb, auto &&fn) const {
        ActiveMeshlets.ForEach(mb.PrimitiveRoot,[&](uint32_t id) { fn(id,Primitives.Get({id,1u})[0]); });
    }
    uint32_t ClusterGroupCount(const MeshBuffers &mb) const { return ActiveMeshlets.Count(mb.GroupRoot); }
    uint32_t PrimitiveRoute(const MeshBuffers &mb, uint32_t source_primitive) const {
        return source_primitive < mb.PrimitiveRoutes.Count ? PrimitiveRoutes.Get({mb.PrimitiveRoutes.Offset + source_primitive, 1u})[0] : InvalidOffset;
    }
    // Grows the owner's routes to cover `count` source primitives, keeping its routes.
    void ReservePrimitiveRoutes(MeshBuffers &, uint32_t count);
    // The element blocks holding the owner's meshlet owner payloads, which each hold an element of a live finest meshlet.
    std::vector<uint32_t> MeshletOwnerBlocks(const MeshBuffers &) const;
    void ForEachLodNode(const MeshBuffers &mb, auto &&fn) const {
        ActiveMeshlets.ForEach(mb.NodeRoot,[&](uint32_t id) { fn(id,LodNodes.Get({id,1u})[0]); });
    }

    // The render ranges of each mesh record, present from its first sync until the record is released.
    std::vector<std::optional<MeshBuffers>> Meshes;
    MeshBuffers &EmplaceMesh(uint32_t store_id, SlottedRange vertices);
    auto &MeshOf(this auto &self, uint32_t store_id) {
        if constexpr (!std::is_const_v<std::remove_reference_t<decltype(self)>>)
            if (self.MeshHistory) self.MeshHistory->Write(store_id,1u);
        return *self.Meshes.at(store_id);
    }
    auto *TryMeshOf(this auto &self, uint32_t store_id) {
        return store_id < self.Meshes.size() && self.Meshes[store_id] ? &self.MeshOf(store_id) : nullptr;
    }
    std::unique_ptr<store::Records> MeshHistory;
    std::unique_ptr<store::Records> LodDepthHistory;
    void ReleaseMesh(uint32_t store_id);

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
    // Mirrors every Meshlets allocation, so an edit's new clusters extend it by their own count.
    BufferArena<MeshletSpatialNode> MeshletSpatialNodes{Ctx,SlotType::Buffer};
    MeshletIndex ActiveMeshlets{Ctx};
    BufferArena<uint32_t> MeshletTriangleIds;
    BufferArena<uint32_t> MeshletVertexCorners;
    BufferArena<uint8_t> MeshletLocalTriangles;
    // The cluster LOD DAG: one group per simplification step, and the selection forest over them.
    BufferArena<ClusterGroup> ClusterGroups;
    BufferArena<LodNode> LodNodes;
    // Reverse edit dependencies mirror canonical record addresses.
    // Their lifetime follows Meshlets/LodNodes.
    // Drawing never reads these arenas.
    BufferArena<uint32_t> MeshletLodLeaves{Ctx,SlotType::Buffer}, LodParents{Ctx,SlotType::Buffer};
    // A group's inputs and proxies occupy packed runs.
    // Group addresses own the ranges.
    // The draw records carry no dependency fields.
    BufferArena<ClusterGroupLinks> GroupLinks{Ctx,SlotType::Buffer};
    BufferArena<uint32_t> GroupClusterIds{Ctx,SlotType::Buffer};
    BufferArena<PrimitiveRecord> Primitives;
    // Each render owner's run of render primitive handles, indexed by source primitive and InvalidOffset where absent.
    BufferArena<uint32_t> PrimitiveRoutes{Ctx,SlotType::Buffer};
    BufferArena<MeshRecord> MeshRecords;
    mtl::Buffer GpuInstanceSlots;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, SlotType::MorphWeightBuffer};
    InstanceArena Instances;

    mtl::Buffer MeshletWorkRanges, MeshletWorkBlocks, MeshletWorkState, MeshletWorkDispatchArgs;
    // Span-tree traversal alternates frontiers and stores each level's size, block prefixes, and indirect arguments.
    std::array<mtl::Buffer, 2> LodFrontiers;
    mtl::Buffer LodFrontierStates, LodFrontierBlockStates, LodExpandArgs;
    MeshletCullOutput SceneCull{Ctx}, EditCull{Ctx};
    mtl::Buffer MeshletClassifications, MeshletCullBlocks;
    // Coarse clusters the last cull's cut selected, which the classification accumulates.
    mtl::Buffer MeshletCoarseCount;
    // Persistent procedural line jobs, deterministically compacted into one indirect submission.
    mtl::Buffer OverlayJobs, OverlayJobBlocks, VisibleOverlayJobs, OverlayJobDispatchArgs;
    // Live LOD nodes and meshlets over the drawing instances, which bound each cull's traversal and work.
    uint64_t LodNodeCount{0};
    uint64_t MeshletInstanceCount{0};
    // Maximum traversal depth among resident mesh span trees.
    uint32_t MeshletLodDepth{0};
    uint32_t MeshletTopologyMask{0};

    // Maintained totals for culls restricted to one instance flag.
    struct MeshletFlagWork {
        uint64_t Nodes{0}, Meshlets{0};
    };
    // One entry per MeshletInstanceFlag bit, indexed by that bit's position.
    static constexpr size_t MeshletInstanceFlagCount = std::bit_width(uint32_t(MeshletInstanceFlag::EdgeOverlay));
    std::array<MeshletFlagWork, MeshletInstanceFlagCount> MeshletFlagWorkByBit{};

    MeshletFlagWork &FlagWork(uint32_t flag) { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }
    const MeshletFlagWork &FlagWork(uint32_t flag) const { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }
    // The flags whose work totals count the meshlet work of their drawing instances.
    static constexpr uint32_t CountedMeshletFlags = ((1u << MeshletInstanceFlagCount) - 1u) &
        ~(uint32_t(MeshletInstanceFlag::LodPinFinest) | uint32_t(MeshletInstanceFlag::OverlayOnly));

    static constexpr uint32_t MeshletDispatchChunkSize{65'535};
    static constexpr uint32_t MeshletCullBlockSize{1024};
    static constexpr uint32_t MeshletRouteCount{uint32_t(MeshletRoute::Count)};
    static constexpr uint32_t OverlayJobBlockSize{256};

    void SetOverlayJobs(std::span<const OverlayJob> jobs);

    void EnsureMeshletVisibilityCapacity(
        MeshletCullOutput &, uint64_t visible_count, uint64_t work_node_count, uint64_t work_meshlet_count
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
    mtl::Buffer Lights;
    // Light buffer indices freed by destroyed lights, compacted by the next event pass.
    std::vector<uint32_t> PendingLightRemovals;
    mtl::Buffer Materials;

    RenderView FrameView{};
    // SceneViewUBO stores the live state at instance zero and one aligned instance per blur step.
    mtl::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;

    // One entry per run of mesh instance slots sharing a deform state.
    mtl::Buffer BoundsReduceEntries{Ctx, 0, SlotType::BoundsEntryBuffer};
    // (entry index, canonical node key within its level).
    // Leaves precede parents.
    mtl::Buffer BoundsTiles{Ctx, 0, SlotType::Buffer};
    std::array<uint32_t,VertexBoundsLevels> BoundsFirstTiles{};
    VertexBoundsStore VertexBounds{Ctx};
    // (entry index, canonical block) per normal-derive threadgroup, face blocks in a leading prefix.
    mtl::Buffer DeriveTiles{Ctx, 0, SlotType::Buffer};
    // Current-pose positions keyed by canonical vertex in each pose namespace.
    PoseAttributeStore<vec3> PosedPositions{Ctx};
    // (posed entry, global meshlet) per posed-meshlet bounds threadgroup, plus its local-space AABB output.
    mtl::Buffer PosedMeshletBoundsJobs{Ctx, 0, SlotType::Buffer};
    PoseAttributeStore<AABB> PosedMeshletBounds{Ctx};
    // One entry per normal-derive dispatch item.
    // Contains one entry per posed triangle range or one per mesh during base derivation.
    mtl::Buffer NormalDeriveEntries{Ctx, 0, SlotType::Buffer};
    // Derived normals keyed by canonical vertex, sector record, and face, respectively.
    PoseAttributeStore<vec3> PosedVertexNormals{Ctx};
    PoseAttributeStore<vec3> PosedSectors{Ctx};
    PoseAttributeStore<vec3> PosedFaceNormals{Ctx};
    // Weight-summed authored morph normal deltas keyed by canonical vertex, present for authored morph poses.
    PoseAttributeStore<vec3> PosedMorphNormalDeltas{Ctx};
    // Group counts of the posed prelude's passes, in recorded order (their arg slot order in PreludeDispatchArgs).
    // Set when persistent scene descriptors refresh.
    struct PreludeGroups {
        static constexpr uint32_t PassCount{7};
        uint32_t PosePrepass{0}, PosedMeshletBounds{0}, DeriveFaces{0}, DeriveGather{0};
        std::array<uint32_t,3> BoundsCombine{};

        // Empty entries still dispatch their root to publish neutral bounds.
        bool HasWork() const { return PosePrepass > 0 || PosedMeshletBounds > 0 || DeriveFaces > 0 || BoundsCombine[2] > 0; }
    };
    PreludeGroups Prelude{};
    // Stores recorded group counts or zeros for unchanged deform inputs.
    mtl::Buffer PreludeDispatchArgs;
    // A deform input was written since the last submit wrote live prelude counts.
    // Deform inputs are morph weights, armature poses, transform gestures, geometry edits, and scene refreshes.
    bool PreludeStale{true};
    // Tracks visibility or material changes that can reveal geometry without requiring the posed prelude.
    bool MeshletOcclusionStale{true};

    // Visibility IDs index the visible list and require matching cull and raster generations for decoding.
    uint32_t MeshletVisibleGeneration{0};
    // The cull generation the visibility image was rasterized against.
    uint32_t VisibilityGeneration{InvalidOffset};

    mtl::Buffer ObjectPickKeys, ObjectPickSeenBitset, ObjectBoxBitset;
    uint32_t ObjectPickEpochTag{}; // Zero clears the persistent keys before the first pick and after wraparound.
    mtl::Buffer ElementPickKey, ElementPickId;
    BufferArena<uint32_t> GeometryWork{Ctx, SlotType::Buffer};
    mtl::Buffer GeometryNormalEntries{Ctx, 0, SlotType::Buffer};
    // Canonical triangle, edge, and vertex handles map to global finest meshlet IDs.
    std::array<ElementAttribute<uint32_t>,3> ElementMeshlets{{{Ctx,SlotType::Buffer},{Ctx,SlotType::Buffer},{Ctx,SlotType::Buffer}}};
    mtl::Buffer WireCoverageBuffer{Ctx, 0, SlotType::Buffer};
};
