#pragma once

#include "AABB.h"
#include "gpu/ElementPickCandidate.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/SelectionCounters.h"
#include "gpu/SelectionNode.h"
#include "gpu/Transform.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "render/MeshBuffers.h"
#include "render/PickConstants.h"
#include "vulkan/BufferArena.h"

#include <algorithm>
#include <numeric>

struct DrawBufferPair {
    mvk::Buffer DrawData;
    // Also a storage buffer, so the frustum cull pass can edit instance counts in place.
    mvk::Buffer Indirect;
    // Maps gl_InstanceIndex to a dense DrawData index. Identity when nothing is culled.
    mvk::Buffer VisibleIndices;
    // One CullEntry per DrawData element, for the frustum cull pass.
    mvk::Buffer CullEntries;
    DrawBufferPair(mvk::BufferContext &ctx)
        : DrawData(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer),
          Indirect(ctx, 0, vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer),
          VisibleIndices(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer),
          CullEntries(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer) {}
};

// Shared arena for per-instance GPU data (transforms, object IDs, instance states, local-space bounds).
// Uses a single RangeAllocator so all buffers share the same instance offsets.
struct InstanceArena {
    InstanceArena(mvk::BufferContext &ctx)
        : TransformBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer),
          ObjectIdBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer),
          StateBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::InstanceStateBuffer),
          BoundsBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer) {}

    Range Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;
        const vk::DeviceSize end = range.Offset + range.Count;
        EnsureCapacity(end);
        return range;
    }

    void Free(Range range) { Allocator.Free(range); }

    // Compact-erase: shift [global_index+1, range_end) down by 1 in all buffers.
    void CompactErase(uint32_t global_index, uint32_t range_end) {
        const auto count = range_end - global_index - 1;
        if (count == 0) return;
        ForEachBuffer([&](mvk::Buffer &buf, size_t sz) {
            buf.Move(vk::DeviceSize(global_index + 1) * sz, vk::DeviceSize(global_index) * sz, count * sz);
        });
    }

    // Copy `count` elements from src_offset to dst_offset across all buffers.
    void CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count) {
        if (count == 0 || src_offset == dst_offset) return;
        ForEachBuffer([&](mvk::Buffer &buf, size_t sz) {
            buf.Move(vk::DeviceSize(src_offset) * sz, vk::DeviceSize(dst_offset) * sz, count * sz);
        });
    }

    // Pre-reserve capacity for `count` additional instances to avoid per-Allocate growth.
    void ReserveAdditional(uint32_t count) { EnsureCapacity(vk::DeviceSize(Allocator.HighWaterMark()) + count); }

    void UpdateState(uint32_t index, uint8_t state) { StateBuffer.Update(as_bytes(state), vk::DeviceSize(index) * sizeof(uint8_t)); }
    void UpdateBounds(uint32_t index, const AABB &bounds) { BoundsBuffer.Update(as_bytes(bounds), vk::DeviceSize(index) * sizeof(AABB)); }
    const AABB &GetBounds(uint32_t index) const { return reinterpret_cast<const AABB *>(BoundsBuffer.GetMappedData().data())[index]; }
    std::span<AABB> GetMutableBounds(Range range) const {
        auto mapped = BoundsBuffer.GetMutableRange(vk::DeviceSize(range.Offset) * sizeof(AABB), vk::DeviceSize(range.Count) * sizeof(AABB));
        return {reinterpret_cast<AABB *>(mapped.data()), range.Count};
    }
    std::span<uint8_t> GetMutableStates() const { return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), StateBuffer.UsedSize}; }
    std::span<Transform> GetMutableTransforms() const {
        auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<Transform *>(mapped.data()), mapped.size() / sizeof(Transform)};
    }

    // Reset to empty: used size and allocator go to zero, keeping the GPU allocations for reuse.
    void Reset() {
        Allocator = {};
        ForEachBuffer([](mvk::Buffer &buf, size_t) { buf.UsedSize = 0; });
    }

    mvk::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer, BoundsBuffer;

private:
    void ForEachBuffer(auto &&fn) {
        fn(TransformBuffer, sizeof(Transform));
        fn(ObjectIdBuffer, sizeof(uint32_t));
        fn(StateBuffer, sizeof(uint8_t));
        fn(BoundsBuffer, sizeof(AABB));
    }

    void EnsureCapacity(vk::DeviceSize end) {
        ForEachBuffer([end](mvk::Buffer &buf, size_t sz) {
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

    // The view UBO instance stride, aligned for dynamic uniform offsets.
    static vk::DeviceSize ViewUboStride(vk::PhysicalDevice pd) {
        const auto align = pd.getProperties().limits.minUniformBufferOffsetAlignment;
        return (sizeof(::SceneViewUBO) + align - 1) / align * align;
    }
    static constexpr uint32_t MaxSelectableElements{10'000'000};
    static constexpr uint32_t ObjectPickBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t SelectionBitsetWords{(MaxSelectableElements + 31) / 32};
    static constexpr uint32_t ElementPickGroupSize{256};
    static constexpr uint32_t ElementPickGroupCount{(ElementPickPixelCount + ElementPickGroupSize - 1) / ElementPickGroupSize};
    static constexpr uint32_t SelectionNodesPerPixel{10};
    static constexpr uint32_t MaxSelectionNodeBytes{64 * 1024 * 1024};

    GpuBuffers(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
        : Ctx{pd, d, instance, slots},
          VertexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          Instances{Ctx},
          Lights{Ctx, sizeof(PunctualLight), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::LightBuffer},
          Materials{Ctx, sizeof(PBRMaterial), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MaterialBuffer},
          SceneViewUBO{Ctx, ViewUboStride(pd) * (MaxBlurSteps + 1), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::SceneViewUBO},
          ViewportThemeUBO{Ctx, sizeof(ViewportTheme), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::ViewportThemeUBO},
          WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::WorkspaceLightsUBO},
          SceneViewUboStride{ViewUboStride(pd)},
          RenderDraw{Ctx}, SelectionDraw{Ctx},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounter{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickKeys{Ctx, MaxSelectableObjects * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickSeenBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          SelectionBitset{Ctx, SelectionBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          MotionBlurTileIndirection{Ctx, 0, mvk::MemoryUsage::GpuOnly, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst},
          ElementPickCandidates{Ctx, ElementPickGroupCount * sizeof(ElementPickCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          IdentityIndexBuffer{Ctx, 0, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndexBuffer} {
        // The dynamic-offset binding describes one instance. Offsets select among the ring's instances.
        SceneViewUBO.DescriptorRange = sizeof(::SceneViewUBO);
        SceneViewUBO.UpdateDescriptor();
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
        for (auto &[_, rb] : buffers.NormalIndicators) Release(rb);
        buffers.NormalIndicators.clear();
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
    void ResizeMotionBlurTileIndirection(vk::Extent2D tile_extent) {
        MotionBlurTileIndirection.SetCount(2 * tile_extent.width * tile_extent.height);
    }

    void ResizeSelectionNodeBuffer(vk::Extent2D extent) {
        const uint64_t pixels = uint64_t(extent.width) * extent.height;
        const uint64_t desired_nodes = pixels == 0 ? 1 : pixels * SelectionNodesPerPixel;
        const uint64_t max_nodes = std::max<uint64_t>(1, MaxSelectionNodeBytes / sizeof(SelectionNode));
        const uint32_t final_count = std::min<uint64_t>({desired_nodes, max_nodes, std::numeric_limits<uint32_t>::max()});
        if (final_count != SelectionNodeCapacity) {
            SelectionNodeCapacity = final_count;
            SelectionNodeBuffer = {Ctx, SelectionNodeCapacity * sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer};
        }
    }

    // Empty the scene arenas on clear so the next load's derived handles (BufferIndex, InstanceRange, morph/bone
    // ranges) rebuild from a clean baseline rather than on a prior scene's leftover residue.
    void ResetSceneArenas() {
        VertexBuffer.Reset();
        FaceIndexBuffer.Reset();
        EdgeIndexBuffer.Reset();
        VertexIndexBuffer.Reset();
        ArmatureDeformBuffer.Reset();
        MorphWeightBuffer.Reset();
        VertexClassBuffer.Reset();
        Instances.Reset();
    }

    void EnsureIdentityIndexBuffer(uint32_t count) {
        if (count <= IdentityIndexCount) return;
        std::vector<uint32_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0u);
        IdentityIndexBuffer.Update(as_bytes(indices));
        IdentityIndexCount = count;
    }

    mvk::BufferContext Ctx;

    // Per-scene arenas
    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphWeightBuffer};
    BufferArena<uint8_t> VertexClassBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexClassBuffer};
    InstanceArena Instances;

    // Poses at the shutter's open and close, for motion blur's velocity pass. Each is a whole-buffer
    // copy of its live counterpart, so the per-draw offsets index them unchanged.
    struct VelocityPose {
        VelocityPose(mvk::BufferContext &ctx)
            : Transforms(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer),
              ArmatureDeform(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ArmatureDeformBuffer),
              MorphWeights(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphWeightBuffer) {}

        mvk::Buffer Transforms, ArmatureDeform, MorphWeights;
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

    // Point the view UBO's draw-data and visible-index slots at `pair`'s buffers. GetDrawData reads both.
    void SetSceneViewDrawSlots(const DrawBufferPair &pair) {
        SceneViewUBO.Update(as_bytes(pair.DrawData.Slot), offsetof(::SceneViewUBO, DrawDataSlot));
        SceneViewUBO.Update(as_bytes(pair.VisibleIndices.Slot), offsetof(::SceneViewUBO, VisibleIndexSlot));
    }

    // Copy the live view UBO into ring instance `instance`.
    void SnapshotSceneViewUbo(uint32_t instance) {
        SceneViewUBO.Update(SceneViewUBO.GetMappedData().subspan(0, sizeof(::SceneViewUBO)), SceneViewUboStride * instance);
    }
    void UpdateSceneViewUboField(uint32_t instance, vk::DeviceSize field_offset, std::span<const std::byte> bytes) {
        SceneViewUBO.Update(bytes, SceneViewUboStride * instance + field_offset);
    }
    uint32_t SceneViewUboOffset(uint32_t instance) const { return uint32_t(SceneViewUboStride * instance); }

    // Snapshot the live pose into `dst`. Call once the scene is evaluated at the wanted time.
    void CaptureVelocityPose(VelocityPose &dst) {
        static constexpr auto copy_whole = [](const mvk::Buffer &src, mvk::Buffer &dst) {
            dst.Reserve(src.UsedSize);
            dst.Update(src.GetMappedData().subspan(0, src.UsedSize));
            dst.UsedSize = src.UsedSize;
        };
        copy_whole(Instances.TransformBuffer, dst.Transforms);
        copy_whole(ArmatureDeformBuffer.Buffer, dst.ArmatureDeform);
        copy_whole(MorphWeightBuffer.Buffer, dst.MorphWeights);
        dst.ViewProj = reinterpret_cast<const ::SceneViewUBO *>(SceneViewUBO.GetMappedData().data())->ViewProj;
    }

    // Per-scene resource tables — reset via their own paths (Lights.SetCount(0) / ResetImportedTexturesAndMaterials)
    TypedBuffer<PunctualLight> Lights;
    TypedBuffer<PBRMaterial> Materials;

    // Per-frame uniforms. SceneViewUBO holds MaxBlurSteps+1 instances at SceneViewUboStride,
    // with the live state at instance 0.
    mvk::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;
    vk::DeviceSize SceneViewUboStride;

    // Draw-command buffers
    DrawBufferPair RenderDraw, SelectionDraw;

    // Selection / picking — GPU buffers + host-visible readback
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    TypedBuffer<SelectionCounters> SelectionCounter;
    TypedBuffer<uint32_t> ObjectPickKeys, ObjectPickSeenBitset, SelectionBitset, MotionBlurTileIndirection;
    TypedBuffer<ElementPickCandidate> ElementPickCandidates;

    // Shared identity sequence backing unindexed draws and seeding visible-index remaps. Grown on demand, not scene-scoped.
    mvk::Buffer IdentityIndexBuffer;
    uint32_t IdentityIndexCount{0};
};
