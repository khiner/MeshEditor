#pragma once

inline void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}

// Submit `cb`, block until `fence` signals, and reset the fence.
inline void SubmitAndWait(vk::Queue queue, vk::CommandBuffer cb, vk::Fence fence, vk::Device device, vk::Semaphore signal_semaphore = {}) {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    queue.submit(submit, fence);
    WaitFor(fence, device);
}
