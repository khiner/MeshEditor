#pragma once

#include "gpu/ClickElementCandidate.h"
#include "gpu/ClickResult.h"
#include "gpu/SelectionCounters.h"
#include "gpu/SelectionNode.h"

constexpr uint32_t
    ClickSelectRadiusPx = 50,
    ClickSelectDiameterPx = ClickSelectRadiusPx * 2 + 1,
    ClickSelectPixelCount = ClickSelectDiameterPx * ClickSelectDiameterPx;

struct SceneBuffers {
    static constexpr uint32_t MaxSelectableObjects{100'000};
    static constexpr uint32_t BoxSelectBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t ClickElementGroupSize{256};
    static constexpr uint32_t ClickSelectElementGroupCount{(ClickSelectPixelCount + ClickElementGroupSize - 1) / ClickElementGroupSize};
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
          RenderDrawData{Ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer},
          RenderIndirect{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer},
          SelectionDrawData{Ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::DrawDataBuffer},
          SelectionIndirect{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndirectBuffer},
          IdentityIndexBuffer{Ctx, 1, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eIndexBuffer},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounterBuffer{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickResultBuffer{Ctx, sizeof(ClickResult), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickElementResultBuffer{Ctx, ClickSelectElementGroupCount * sizeof(ClickElementCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          BoxSelectBitsetBuffer{Ctx, BoxSelectBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {}

    const ClickResult &GetClickResult() const { return *reinterpret_cast<const ClickResult *>(ClickResultBuffer.GetData().data()); }
    vk::DescriptorBufferInfo GetBoxSelectBitsetDescriptor() const { return {*BoxSelectBitsetBuffer, 0, BoxSelectBitsetWords * sizeof(uint32_t)}; }

    SlottedBufferRange CreateIndices(std::span<const uint> indices, IndexKind index_kind) {
        auto &index_buffer = GetIndexBuffer(index_kind);
        return {index_buffer.Allocate(indices), index_buffer.Buffer.Slot};
    }
    RenderBuffers CreateRenderBuffers(std::span<const Vertex> vertices, std::span<const uint> indices, IndexKind index_kind) {
        return {VertexBuffer.Allocate(vertices), CreateIndices(indices, index_kind), index_kind};
    }

    void Release(RenderBuffers &buffers) {
        VertexBuffer.Release(buffers.Vertices);
        buffers.Vertices = {};
        GetIndexBuffer(buffers.IndexType).Release(buffers.Indices.Range);
        buffers.Indices.Range = {};
    }

    void Release(MeshBuffers &buffers) {
        FaceIndexBuffer.Release(buffers.FaceIndices.Range);
        buffers.FaceIndices.Range = {};
        EdgeIndexBuffer.Release(buffers.EdgeIndices.Range);
        buffers.EdgeIndices.Range = {};
        VertexIndexBuffer.Release(buffers.VertexIndices.Range);
        buffers.VertexIndices.Range = {};
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

    mvk::BufferContext Ctx;
    BufferArena<Vertex> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    mvk::Buffer SceneViewUBO;
    mvk::Buffer ViewportThemeUBO;
    mvk::Buffer RenderDrawData;
    mvk::Buffer RenderIndirect;
    mvk::Buffer SelectionDrawData;
    mvk::Buffer SelectionIndirect;
    mvk::Buffer IdentityIndexBuffer;
    uint32_t IdentityIndexCount{0};
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    // CPU readback buffers (host-visible)
    mvk::Buffer SelectionCounterBuffer, ClickResultBuffer, ClickElementResultBuffer, BoxSelectBitsetBuffer;
};
