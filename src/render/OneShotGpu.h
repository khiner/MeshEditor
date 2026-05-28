#pragma once

#include <vulkan/vulkan.hpp>

#include <cstdint>

// One-shot GPU sync primitives for synchronous passes (selection compute, element pick,
// glTF load, texture uploads, etc.). Owns its Vk resources directly; the render-pipeline
// command buffers are allocated from `Pool` and freed before this component is destroyed.
struct OneShotGpu {
    vk::UniqueCommandPool Pool;
    vk::UniqueCommandBuffer Cb;
    vk::UniqueFence Fence;
    vk::UniqueSemaphore SelectionReady;
};

inline OneShotGpu MakeOneShotGpu(vk::Device device, uint32_t queue_family) {
    auto pool = device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family});
    auto cb = std::move(device.allocateCommandBuffersUnique({*pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    return {.Pool = std::move(pool), .Cb = std::move(cb), .Fence = device.createFenceUnique({}), .SelectionReady = device.createSemaphoreUnique({})};
}
