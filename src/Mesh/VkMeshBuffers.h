#pragma once

#include "MeshBuffers.h"
#include "Vulkan/VulkanBuffer.h"

struct VulkanContext;

struct VkMeshBuffers {
    VkMeshBuffers(const VulkanContext &vc) : VC(vc) {}
    VkMeshBuffers(const VulkanContext &vc, std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices)
        : VC(vc), Buffers(std::move(vertices), std::move(indices)) {
        CreateOrUpdateBuffers();
    }
    VkMeshBuffers(const VulkanContext &vc, MeshBuffers &&buffers)
        : VC(vc), Buffers(std::move(buffers)) {
        CreateOrUpdateBuffers();
    }
    ~VkMeshBuffers() = default;

    const std::vector<Vertex3D> &GetVertices() const { return Buffers.Vertices; }
    const std::vector<uint> &GetIndices() const { return Buffers.Indices; }

    inline void Set(MeshBuffers &&buffers) {
        Buffers = std::move(buffers);
        CreateOrUpdateBuffers();
    }

    void CreateOrUpdateBuffers();
    void Bind(vk::CommandBuffer) const;

    const VulkanContext &VC;
    MeshBuffers Buffers;
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
};
