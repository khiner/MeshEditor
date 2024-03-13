#pragma once

#include <vulkan/vulkan.hpp>

// todo next up (see https://github.com/khiner/MeshEditor/commit/91c2e6cf9ebf16a3ef31baa202688cd627416ce5)
// TODO (on main) first:
// - use single contiguous vulkan buffers instead of per-mesh vulkan buffers, and
//   manage outside of `Registry`.
// - `Registry` entities have contiguous `uint`-wrapping ID components indexing into the vulkan buffers they need.
// - E.g. `struct MeshVertexIndex { uint i; };`

struct VulkanBuffer {
    vk::BufferUsageFlags Usage;
    vk::DeviceSize Size{0};

    // GPU buffer.
    vk::UniqueBuffer Buffer{};
    vk::UniqueDeviceMemory Memory{};

    // Host staging buffer, used to transfer data to the GPU.
    vk::UniqueBuffer StagingBuffer{};
    vk::UniqueDeviceMemory StagingMemory{};
};
