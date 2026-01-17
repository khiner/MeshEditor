#pragma once

#include <cstddef>
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
    ObjectIdBuffer,
    FaceNormalBuffer,
    DrawDataBuffer,
    Count
};
constexpr size_t SlotTypeCount{static_cast<size_t>(SlotType::Count)};

struct TypedSlot {
    SlotType Type;
    uint32_t Slot;
    bool operator==(const TypedSlot &) const = default;
};
