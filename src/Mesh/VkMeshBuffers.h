#pragma once

#include "vulkan/VulkanBuffer.h"

struct MeshBuffers;
struct VulkanContext;

struct VkMeshBuffers {
    VkMeshBuffers();
    VkMeshBuffers(VkMeshBuffers &&) = default;
    VkMeshBuffers(const VulkanContext &, MeshBuffers &&);
    ~VkMeshBuffers();

    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};

    void Set(const VulkanContext &, MeshBuffers &&);
    void Bind(vk::CommandBuffer) const;
    void Draw(vk::CommandBuffer, uint instance_count = 1) const;
};
