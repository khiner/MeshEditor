#pragma once
#include "viewport/ViewportSchema.h"

// Resize waits for the live ImGui command's serial before replacing its sampled image.
struct ViewportConsumerFence {
    uint64_t Value{};
};
