#pragma once

#include <cstdint>

// Component on a light entity. Indexes into the GpuBuffers::Lights buffer (canonical PunctualLight storage).
struct LightIndex {
    uint32_t Value{0};
};
// Index into the shared lights buffer.
