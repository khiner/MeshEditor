#pragma once

#include <vulkan/vulkan.hpp>

// The live ImGui frame fence sampling the viewport color image; the resize path waits on it before
// recreating resources. Empty during replay, where the wait is a no-op.
struct ViewportConsumerFence {
    vk::Fence Value{};
};
