#pragma once

#include "vulkan/VulkanBuffer.h"

struct MeshBuffers;
struct VulkanContext;

struct VkMeshBuffers {
    VkMeshBuffers(const VulkanContext &);
    VkMeshBuffers(const VulkanContext &, MeshBuffers &&);
    ~VkMeshBuffers();

    const VulkanContext &VC;
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};

    void Set(MeshBuffers &&);
    void Bind(vk::CommandBuffer) const;
    void Draw(vk::CommandBuffer, uint instance_count = 1) const;
};
