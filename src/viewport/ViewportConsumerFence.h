#pragma once
#include "viewport/ViewportSchema.h"

namespace MTL {
class CommandBuffer;
} // namespace MTL

// Resize waits for the live ImGui command buffer before replacing its sampled image.
struct ViewportConsumerFence {
    MTL::CommandBuffer *Value{nullptr};
};
