#pragma once

#include "metal/MetalCpp.h"

// Resize waits for the live ImGui command buffer before replacing its sampled image.
struct ViewportConsumerFence {
    MTL::CommandBuffer *Value{nullptr};
};
