#pragma once

#include "vulkan/VulkanBuffer.h"

struct VkMeshBuffers {
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
};
