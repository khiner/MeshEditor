#pragma once

#include "gpu/BoundsEntry.h"
#include "gpu/InstanceRecord.h"
#include "gpu/LightRecord.h"
#include "gpu/LodFrontierBlockState.h"
#include "gpu/LodFrontierEntry.h"
#include "gpu/LodFrontierState.h"
#include "gpu/MeshDispatchArgs.h"
#include "gpu/MeshletCullBlockState.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRouteState.h"
#include "gpu/MeshletWorkRange.h"
#include "gpu/MeshletWorkState.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/OverlayJob.h"
#include "gpu/PBRMaterial.h"
#include "gpu/Transform.h"
#include "gpu/ViewportTheme.h"
#include "gpu/VisibleMeshlet.h"
#include "gpu/WorkspaceLights.h"
#include "render/PoseAttributeStore.h"
#include "render/VertexBoundsStore.h"
#include "viewport/RenderView.h"

// Per-instance GPU data behind one RangeAllocator, so every buffer shares the same instance offsets.
// A slot without an instance contains a zero state, so a pass over every slot skips it.
struct InstanceArena {
    InstanceArena(mtl::BufferContext &ctx);

    Range Allocate(uint32_t count);
    void Free(Range range) {
        ClearStates(range);
        Allocator.Free(range);
    }
    void Free(std::vector<Range> ranges) {
        for (const auto range : ranges) ClearStates(range);
        Allocator.Free(std::move(ranges));
    }

    template<typename Index = std::identity>
    void CompactErase(Range active, const auto &indices, Index index = {}) {
        ForEachSurvivorRun(active, indices, [&](uint32_t from, uint32_t to, uint32_t count) { CopyInstances(from, to, count); }, index);
        const auto erased = uint32_t(std::ranges::size(indices));
        ClearStates({active.Offset + active.Count - erased, erased});
    }
    void CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count);
    void UpdateState(uint32_t index, uint8_t state) { StateBuffer.Update(as_bytes(state), uint64_t(index) * sizeof(uint8_t)); }
    const AABB &GetBounds(uint32_t index) const { return reinterpret_cast<const AABB *>(BoundsBuffer.Contents().data())[index]; }
    std::span<AABB> GetMutableBounds(Range range) const { return BoundsBuffer.GetMutableSpan<AABB>(range); }
    std::span<uint8_t> GetMutableStates() const { return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), StateBuffer.UsedSize}; }
    std::span<Transform> GetMutableTransforms() const {
        const auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<Transform *>(mapped.data()), mapped.size() / sizeof(Transform)};
    }

    // Zero the used sizes and the allocator, keeping the GPU allocations for reuse.
    void Reset();

    mtl::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer, BoundsBuffer, RecordBuffer;

private:
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void ClearStates(Range range) { std::ranges::fill(StateBuffer.GetMutableSpan<uint8_t>(range), uint8_t{0}); }
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

struct RenderArenas;

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
    // The mesh store's render arenas, whose slots the culls and draws bind.
    const RenderArenas *Render{};

    mtl::BufferContext Ctx;

    // Every instance slot in drawing order, by descending object ID, so coplanar ties resolve the same way across loads.
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
    MeshletCullOutput SilhouetteCull{Ctx}; // Outlined surfaces the visibility image cannot resolve.
    // Coarse clusters the last cull's cut selected, which the classification accumulates.
    mtl::Buffer MeshletCoarseCount;
    // Persistent procedural line jobs, deterministically compacted into one indirect submission.
    mtl::Buffer OverlayJobs, OverlayJobBlocks, VisibleOverlayJobs, OverlayJobDispatchArgs;
    // Live LOD nodes and meshlets over the drawing instances, which bound each cull's traversal and work, summed over the mesh tallies.
    uint64_t LodNodeCount{0};
    uint64_t MeshletInstanceCount{0};
    // Maximum traversal depth among resident mesh span trees, the deepest tally.
    uint32_t MeshletLodDepth{0};
    // The tallied meshes drawing meshlets of each render topology, by topology bit: triangles, lines, points.
    std::array<uint32_t, 3> TopologyMeshes{};
    // The tallied meshes whose edit overlay draws sharp edges.
    uint32_t SharpEditMeshes{0};
    bool DrawsNonTriangles() const { return TopologyMeshes[1] != 0u || TopologyMeshes[2] != 0u; }

    // Maintained totals for culls restricted to one instance flag.
    struct MeshletFlagWork {
        uint64_t Nodes{0}, Meshlets{0};
    };
    // One entry per MeshletInstanceFlag bit, indexed by that bit's position.
    static constexpr size_t MeshletInstanceFlagCount = std::bit_width(uint32_t(MeshletInstanceFlag::SilhouetteEligible));
    std::array<MeshletFlagWork, MeshletInstanceFlagCount> MeshletFlagWorkByBit{};

    MeshletFlagWork &FlagWork(uint32_t flag) { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }
    const MeshletFlagWork &FlagWork(uint32_t flag) const { return MeshletFlagWorkByBit[std::countr_zero(flag)]; }
    // The mesh flags whose totals count the meshlet work of the mesh's instances.
    // The settle pass totals Silhouette over the selected instances.
    static constexpr uint32_t CountedMeshletFlags = ((1u << MeshletInstanceFlagCount) - 1u) &
        ~(uint32_t(MeshletInstanceFlag::Silhouette) | uint32_t(MeshletInstanceFlag::LodPinFinest) |
          uint32_t(MeshletInstanceFlag::OverlayOnly) | uint32_t(MeshletInstanceFlag::SilhouetteEligible));
    // Count visible meshlet work by record flags and topology, including enabled sharp-edge overlays.
    struct MeshFlagTally {
        uint32_t Flags{0}, Instances{0}, Nodes{0}, Meshlets{0}, Depth{0}, Topologies{0};
        bool SharpEdges{};
    };
    // Each mesh record's tally, by store id.
    std::vector<MeshFlagTally> FlagTallies;
    // Moves the mesh record's contribution to the scene and flag totals to `tally`.
    void Retally(uint32_t store_id, MeshFlagTally tally);

    static constexpr uint32_t MeshletDispatchChunkSize{65'535};
    static constexpr uint32_t MeshletCullBlockSize{1024};
    static constexpr uint32_t MeshletRouteCount{uint32_t(MeshletRoute::Count)};
    static constexpr uint32_t OverlayJobBlockSize{256};

    // Sizes the overlay job list and its cull scratch to `count` jobs, and returns the list for writing.
    std::span<OverlayJob> ResizeOverlayJobs(uint32_t count);

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

    mtl::Buffer Lights;
    // Light buffer indices freed by destroyed lights, compacted by the next event pass.
    std::vector<uint32_t> PendingLightRemovals;
    mtl::Buffer Materials;

    RenderView FrameView{};
    // SceneViewUBO stores the live state at instance zero and one aligned instance per blur step.
    mtl::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;

    // One entry per run of mesh instance slots sharing a deform state, in each laid-out mesh's range.
    BufferArena<BoundsEntry> BoundsReduceEntries{Ctx, SlotType::BoundsEntryBuffer};
    VertexBoundsStore VertexBounds{Ctx};
    // Current-pose positions keyed by canonical vertex in each pose namespace.
    PoseAttributeStore<vec3> PosedPositions{Ctx};
    // Local-space posed meshlet AABBs.
    PoseAttributeStore<AABB> PosedMeshletBounds{Ctx};
    // One entry per posed bounds entry whose normals derive, in each laid-out mesh's range.
    BufferArena<NormalDeriveEntry> NormalDeriveEntries{Ctx, SlotType::Buffer};
    // Each laid-out mesh's prelude tile words, which only the host reads to gather the tiles of the meshes a prelude recomputes.
    BufferArena<uint32_t> PreludeTiles{Ctx, SlotType::Buffer};
    // Derived normals keyed by canonical vertex, sector record, and face, respectively.
    PoseAttributeStore<vec3> PosedVertexNormals{Ctx};
    PoseAttributeStore<vec3> PosedSectors{Ctx};
    PoseAttributeStore<vec3> PosedFaceNormals{Ctx};
    // Weight-summed authored morph normal deltas keyed by canonical vertex, present for authored morph poses.
    PoseAttributeStore<vec3> PosedMorphNormalDeltas{Ctx};
    // Group prelude tiles by level.
    // Bounds tiles contain entry and node keys; derive tiles contain entry and block keys, with face blocks first.
    mtl::Buffer SparseBoundsTiles{Ctx, 0, SlotType::Buffer}, SparseDeriveTiles{Ctx, 0, SlotType::Buffer}, SparsePosedMeshletBoundsJobs{Ctx, 0, SlotType::Buffer};
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
    mtl::Buffer WireCoverageBuffer{Ctx, 0, SlotType::Buffer};
};

// Resets every per-scene arena, table, and tally to the scene-load baseline, keeping the GPU allocations.
void Reset(GpuBuffers &, state::Scene &);
