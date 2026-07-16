#pragma once

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
    mvk::Buffer Indirect;
    DrawBufferPair(mvk::BufferContext &ctx)
        : DrawData(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer),
          Indirect(ctx, 0, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer) {}
};

// Shared arena for per-instance GPU data (transforms, object IDs, instance states).
// Uses a single RangeAllocator so all 3 buffers share the same instance offsets.
struct InstanceArena {
    InstanceArena(mvk::BufferContext &ctx)
        : TransformBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer),
          ObjectIdBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer),
          StateBuffer(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::InstanceStateBuffer) {}

    Range Allocate(uint32_t count) {
        const auto range = Allocator.Allocate(count);
        if (range.Count == 0) return range;
        const vk::DeviceSize end = range.Offset + range.Count;
        EnsureCapacity(end);
        return range;
    }

    void Free(Range range) { Allocator.Free(range); }

    // Compact-erase: shift [global_index+1, range_end) down by 1 in all 3 buffers.
    void CompactErase(uint32_t global_index, uint32_t range_end) {
        const auto count = range_end - global_index - 1;
        if (count == 0) return;
        ForEachBuffer([&](mvk::Buffer &buf, size_t sz) {
            buf.Move(vk::DeviceSize(global_index + 1) * sz, vk::DeviceSize(global_index) * sz, count * sz);
        });
    }

    // Copy `count` elements from src_offset to dst_offset across all 3 buffers.
    void CopyInstances(uint32_t src_offset, uint32_t dst_offset, uint32_t count) {
        if (count == 0 || src_offset == dst_offset) return;
        ForEachBuffer([&](mvk::Buffer &buf, size_t sz) {
            buf.Move(vk::DeviceSize(src_offset) * sz, vk::DeviceSize(dst_offset) * sz, count * sz);
        });
    }

    // Pre-reserve capacity for `count` additional instances to avoid per-Allocate growth.
    void ReserveAdditional(uint32_t count) {
        EnsureCapacity(vk::DeviceSize(Allocator.HighWaterMark()) + count);
    }

    void UpdateState(uint32_t index, uint8_t state) { StateBuffer.Update(as_bytes(state), vk::DeviceSize(index) * sizeof(uint8_t)); }
    std::span<uint8_t> GetMutableStates() const {
        return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), StateBuffer.UsedSize};
    }
    std::span<Transform> GetMutableTransforms() const {
        auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<Transform *>(mapped.data()), mapped.size() / sizeof(Transform)};
    }

    // Reset to empty: used size and allocator go to zero, keeping the GPU allocations for reuse.
    void Reset() {
        Allocator = {};
        ForEachBuffer([](mvk::Buffer &buf, size_t) { buf.UsedSize = 0; });
    }

    mvk::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer;

private:
    void ForEachBuffer(auto &&fn) {
        fn(TransformBuffer, sizeof(Transform));
        fn(ObjectIdBuffer, sizeof(uint32_t));
        fn(StateBuffer, sizeof(uint8_t));
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
    static constexpr uint32_t MaxSelectableElements{10'000'000};
    static constexpr uint32_t ObjectPickBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t SelectionBitsetWords{(MaxSelectableElements + 31) / 32};
    static constexpr uint32_t ElementPickGroupSize{256};
    static constexpr uint32_t ElementPickGroupCount{(ElementPickPixelCount + ElementPickGroupSize - 1) / ElementPickGroupSize};
    static constexpr uint32_t SelectionNodesPerPixel{10};
    static constexpr uint32_t MaxSelectionNodeBytes{64 * 1024 * 1024};
    // Motion blur's tile indirection table, one entry per tile per motion direction. Tile
    // coordinates pack into 9 bits, so 512 is the widest grid it addresses.
    static constexpr uint32_t MotionBlurMaxTile{512};
    static constexpr uint32_t MotionBlurTileIndirectionWords{2 * MotionBlurMaxTile * MotionBlurMaxTile};

    GpuBuffers(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
        : Ctx{pd, d, instance, slots},
          VertexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          Instances{Ctx},
          Lights{Ctx, sizeof(PunctualLight), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::LightBuffer},
          Materials{Ctx, sizeof(PBRMaterial), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MaterialBuffer},
          SceneViewUBO{Ctx, sizeof(SceneViewUBO), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::SceneViewUBO},
          ViewportThemeUBO{Ctx, sizeof(ViewportTheme), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::ViewportThemeUBO},
          WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::WorkspaceLightsUBO},
          RenderDraw{Ctx}, SelectionDraw{Ctx},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounter{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickKeys{Ctx, MaxSelectableObjects * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickSeenBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          SelectionBitset{Ctx, SelectionBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          MotionBlurTileIndirection{Ctx, MotionBlurTileIndirectionWords * sizeof(uint32_t), mvk::MemoryUsage::GpuOnly, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst},
          ElementPickCandidates{Ctx, ElementPickGroupCount * sizeof(ElementPickCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          IdentityIndexBuffer{Ctx, 0, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndexBuffer} {}

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

    // Per-frame uniforms
    mvk::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;

    // Draw-command buffers
    DrawBufferPair RenderDraw, SelectionDraw;

    // Selection / picking — GPU buffers + host-visible readback
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    TypedBuffer<SelectionCounters> SelectionCounter;
    TypedBuffer<uint32_t> ObjectPickKeys, ObjectPickSeenBitset, SelectionBitset, MotionBlurTileIndirection;
    TypedBuffer<ElementPickCandidate> ElementPickCandidates;

    // Shared identity index buffer. Grown on demand, not scene-scoped.
    mvk::Buffer IdentityIndexBuffer;
    uint32_t IdentityIndexCount{0};
};
