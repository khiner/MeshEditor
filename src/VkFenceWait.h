#pragma once

inline void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}
