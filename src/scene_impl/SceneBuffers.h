#pragma once

#include "gpu/ElementPickCandidate.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/SelectionCounters.h"
#include "gpu/SelectionNode.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "vulkan/BufferArena.h"

#include <numeric>
#include <unordered_map>

using uint = uint32_t;

enum class IndexKind {
    Face,
    Edge,
    Vertex
};

constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct RenderBuffers {
    RenderBuffers(Range vertices, SlottedRange indices, IndexKind index_type)
        : Vertices(vertices), Indices(indices), IndexType(index_type) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers &operator=(RenderBuffers &&) = default;
    RenderBuffers(const RenderBuffers &) = delete;
    RenderBuffers &operator=(const RenderBuffers &) = delete;

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
          SceneViewUBO{Ctx, sizeof(SceneViewUBO), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::SceneViewUBO},
          ViewportThemeUBO{Ctx, sizeof(ViewportTheme), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::ViewportThemeUBO},
          WorkspaceLightsUBO{Ctx, sizeof(WorkspaceLights), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::WorkspaceLightsUBO},
          RenderDrawData{Ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer},
          RenderIndirect{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer},
          SelectionDrawData{Ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer},
          SelectionIndirect{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer},
          LightBuffer{Ctx, sizeof(PunctualLight), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::LightBuffer},
          MaterialBuffer{Ctx, sizeof(PBRMaterial), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MaterialBuffer},
          IdentityIndexBuffer{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndexBuffer},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounterBuffer{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickKeyBuffer{Ctx, MaxSelectableObjects * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ElementPickCandidateBuffer{Ctx, ElementPickGroupCount * sizeof(ElementPickCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ObjectPickSeenBitsetBuffer{Ctx, ObjectPickBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          SelectionBitsetBuffer{Ctx, SelectionBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {}

    vk::DescriptorBufferInfo GetSelectionBitsetDescriptor() const { return {*SelectionBitsetBuffer, 0, SelectionBitsetWords * sizeof(uint32_t)}; }
    vk::DescriptorBufferInfo GetObjectPickSeenBitsetDescriptor() const { return {*ObjectPickSeenBitsetBuffer, 0, ObjectPickBitsetWords * sizeof(uint32_t)}; }

    SlottedRange CreateIndices(std::span<const uint> indices, IndexKind index_kind) {
        auto &index_buffer = GetIndexBuffer(index_kind);
        return {index_buffer.Allocate(indices), index_buffer.Buffer.Slot};
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

    PunctualLight GetLight(uint32_t index) const { return reinterpret_cast<const PunctualLight *>(LightBuffer.GetMappedData().data())[index]; }
    void SetLight(uint32_t index, const PunctualLight &light) { LightBuffer.Update(as_bytes(light), vk::DeviceSize(index) * sizeof(PunctualLight)); }

    mvk::BufferContext Ctx;
    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    mvk::Buffer SceneViewUBO, ViewportThemeUBO, WorkspaceLightsUBO;
    mvk::Buffer RenderDrawData, RenderIndirect;
    mvk::Buffer SelectionDrawData, SelectionIndirect;
    mvk::Buffer LightBuffer, MaterialBuffer;
    mvk::Buffer IdentityIndexBuffer;
    uint32_t IdentityIndexCount{0};
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    BufferArena<mat4> ArmatureDeformBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ArmatureDeformBuffer};
    BufferArena<float> MorphWeightBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphWeightBuffer};
    BufferArena<uint8_t> VertexClassBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexClassBuffer};
    // CPU readback buffers (host-visible)
    mvk::Buffer SelectionCounterBuffer, ObjectPickKeyBuffer, ElementPickCandidateBuffer, ObjectPickSeenBitsetBuffer, SelectionBitsetBuffer;
};
