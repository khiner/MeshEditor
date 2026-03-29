#pragma once

#include "gpu/ElementPickCandidate.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/SelectionCounters.h"
#include "gpu/SelectionNode.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "gpu/WorldTransform.h"
#include "vulkan/BufferArena.h"

#include <numeric>
#include <string>
#include <unordered_map>

using uint = uint32_t;

struct MaterialStore {
    std::vector<std::string> Names;
};

enum class IndexKind {
    Face,
    Edge,
    Vertex
};

constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct DrawBufferPair {
    mvk::Buffer DrawData;
    mvk::Buffer Indirect;
    DrawBufferPair(mvk::BufferContext &ctx)
        : DrawData(ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer),
          Indirect(ctx, 0, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer) {}
};

struct RenderBuffers {
    RenderBuffers(Range vertices, SlottedRange indices, IndexKind index_type)
        : Vertices(vertices), Indices(indices), IndexType(index_type) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers &operator=(RenderBuffers &&) = default;

    Range Vertices;
    SlottedRange Indices;
    IndexKind IndexType;
};

struct BoundingBoxesBuffers {
    RenderBuffers Buffers;
};

struct MeshBuffers {
    MeshBuffers(SlottedRange vertices, SlottedRange face_indices, SlottedRange edge_indices, SlottedRange vertex_indices)
        : Vertices{vertices}, FaceIndices{face_indices}, EdgeIndices{edge_indices}, VertexIndices{vertex_indices} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;
    SlottedRange Vertices;
    SlottedRange FaceIndices, EdgeIndices, VertexIndices;
    std::unordered_map<Element, RenderBuffers> NormalIndicators;
};

// Adjacency indices for bone silhouette edge detection (stored on armature object entities).
struct BoneAdjacencyIndices {
    SlottedRange Indices;
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
    std::span<uint8_t> GetMutableStates() {
        return {reinterpret_cast<uint8_t *>(StateBuffer.GetMutableRange(0, StateBuffer.UsedSize).data()), static_cast<size_t>(StateBuffer.UsedSize)};
    }
    std::span<WorldTransform> GetMutableTransforms() {
        auto mapped = TransformBuffer.GetMutableRange(0, TransformBuffer.UsedSize);
        return {reinterpret_cast<WorldTransform *>(mapped.data()), mapped.size() / sizeof(WorldTransform)};
    }

    mvk::Buffer TransformBuffer, ObjectIdBuffer, StateBuffer;

private:
    void ForEachBuffer(auto &&fn) {
        fn(TransformBuffer, sizeof(WorldTransform));
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

constexpr uint32_t
    ObjectSelectRadiusPx = 15,
    ElementSelectRadiusPx = 50,
    ElementPickDiameterPx = ElementSelectRadiusPx * 2 + 1,
    ElementPickPixelCount = ElementPickDiameterPx * ElementPickDiameterPx;

struct SceneBuffers {
    static constexpr uint32_t MaxSelectableObjects{100'000};
    static constexpr uint32_t MaxSelectableElements{10'000'000};
    static constexpr uint32_t ObjectPickBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t SelectionBitsetWords{(MaxSelectableElements + 31) / 32};
    static constexpr uint32_t ElementPickGroupSize{256};
    static constexpr uint32_t ElementPickGroupCount{(ElementPickPixelCount + ElementPickGroupSize - 1) / ElementPickGroupSize};
    static constexpr uint32_t SelectionNodesPerPixel{10};
    static constexpr uint32_t MaxSelectionNodeBytes{64 * 1024 * 1024};

    SceneBuffers(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
        : Ctx{pd, d, instance, slots},
          VertexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          Instances{Ctx},
          SceneViewUBO{Ctx, sizeof(SceneViewUBO), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::SceneViewUBO},
          ViewportThemeUBO{Ctx, sizeof(ViewportTheme), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::ViewportThemeUBO},
          WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::WorkspaceLightsUBO},
          RenderDraw{Ctx},
          SelectionDraw{Ctx},
          Lights{Ctx, sizeof(PunctualLight), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::LightBuffer},
          Materials{Ctx, sizeof(PBRMaterial), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MaterialBuffer},
          IdentityIndexBuffer{Ctx, 0, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndexBuffer},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounter{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickKeys{Ctx, MaxSelectableObjects * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickSeenBitset{Ctx, ObjectPickBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          SelectionBitset{Ctx, SelectionBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ElementPickCandidates{Ctx, ElementPickGroupCount * sizeof(ElementPickCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {}

    void ReserveAdditionalIndices(uint32_t face, uint32_t edge, uint32_t vertex) {
        FaceIndexBuffer.ReserveAdditional(face);
        EdgeIndexBuffer.ReserveAdditional(edge);
        VertexIndexBuffer.ReserveAdditional(vertex);
    }

    SlottedRange CreateIndices(std::span<const uint> indices, IndexKind index_kind) {
        auto &index_buffer = GetIndexBuffer(index_kind);
        return {index_buffer.Allocate(indices), index_buffer.Buffer.Slot};
    }
    // Allocate an index range without writing data. Returns the range and a mutable span for direct writes.
    std::pair<SlottedRange, std::span<uint32_t>> AllocateIndices(uint32_t count, IndexKind index_kind) {
        auto &buf = GetIndexBuffer(index_kind);
        auto range = buf.Allocate(count);
        return {{range, buf.Buffer.Slot}, buf.GetMutable(range)};
    }
    RenderBuffers CreateRenderBuffers(std::span<const Vertex> vertices, std::span<const uint> indices, IndexKind index_kind) {
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
        const uint32_t final_count = std::min<uint64_t>(std::min(desired_nodes, max_nodes), std::numeric_limits<uint32_t>::max());
        if (final_count == SelectionNodeCapacity) return;
        SelectionNodeCapacity = final_count;
        SelectionNodeBuffer = {Ctx, SelectionNodeCapacity * sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer};
    }

    void EnsureIdentityIndexBuffer(uint32_t count) {
        if (count <= IdentityIndexCount) return;
        std::vector<uint32_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0u);
        IdentityIndexBuffer.Update(as_bytes(indices));
        IdentityIndexCount = count;
    }

    WorkspaceLights &GetWorkspaceLights() { return *reinterpret_cast<WorkspaceLights *>(WorkspaceLightsUBO.GetMappedData().data()); }

    mvk::BufferContext Ctx;
    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    InstanceArena Instances;
    mvk::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;
    DrawBufferPair RenderDraw, SelectionDraw;
    TypedBuffer<PunctualLight> Lights;
    TypedBuffer<PBRMaterial> Materials;
    mvk::Buffer IdentityIndexBuffer;
    uint32_t IdentityIndexCount{0};
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphWeightBuffer};
    BufferArena<uint8_t> VertexClassBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexClassBuffer};
    // CPU readback buffers (host-visible)
    TypedBuffer<SelectionCounters> SelectionCounter;
    TypedBuffer<uint32_t> ObjectPickKeys, ObjectPickSeenBitset, SelectionBitset;
    TypedBuffer<ElementPickCandidate> ElementPickCandidates;
};

inline vk::Extent2D ComputeRenderExtentPx(vk::Extent2D logical_extent, vec2 scale) {
    const auto scaled_dim = [](uint32_t logical, float s) -> uint32_t {
        if (logical == 0u) return 0u;
        return std::max(1u, uint32_t(float(logical) * (s > 0.0f ? s : 1.0f) + 0.5f));
    };
    return {scaled_dim(logical_extent.width, scale.x), scaled_dim(logical_extent.height, scale.y)};
}
