#pragma once

#include <cstdint>

constexpr uint32_t InvalidSlot{~0u};

enum class SlotType : uint8_t {
    Uniform,
    Image,
    Sampler,
    Buffer, // General storage buffers (selection, counters, etc.)
    VertexBuffer,
    IndexBuffer,
    ModelBuffer,
    ObjectIdBuffer
};
constexpr uint8_t SlotTypeCount{8};
