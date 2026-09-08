#pragma once

namespace MTL {
class CommandBuffer;
}

// Resize waits for the live ImGui command buffer before replacing its sampled image.
struct ViewportConsumerFence {
    MTL::CommandBuffer *Value{nullptr};
};
