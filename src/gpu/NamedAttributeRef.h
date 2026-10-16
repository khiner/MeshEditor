#pragma once
#include "gpu/Types.h"

// Value types of named attributes, each stored in its own value arena.
enum class AttributeType : uint32_t {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Int,
    Bool,
};
GPU_CONSTANT uint32_t AttributeTypeCount = 6u;
// The 32-bit words of one value, where a bool takes one bit of its block's mask words.
inline uint32_t AttributeComponents(AttributeType type) {
    switch (type) {
        case AttributeType::Vec2: return 2u;
        case AttributeType::Vec3: return 3u;
        case AttributeType::Vec4: return 4u;
        case AttributeType::Float:
        case AttributeType::Int:
        case AttributeType::Bool: return 1u;
    }
    return 1u;
}

enum class AttributeDomain : uint32_t {
    Vertex,
    Edge,
    Face,
    Corner,
};

// A named attribute's values at its record's handles in its domain: its value arena's slot and the index of handle 0, a bit index for a bool.
struct NamedAttributeRef {
    uint32_t ValuesSlot DEFAULT(InvalidSlot);
    uint32_t Offset DEFAULT();
};

// One named attribute a topology job carries from its source elements to its output elements, in its domain's run of the job's records.
// A source without the attribute has no values slot.
// Default holds the value bits of output elements without a source value.
struct TopologyAttribute {
    AttributeType Type DEFAULT();
    NamedAttributeRef Source DEFAULT(), Destination DEFAULT();
    GpuArray<uint32_t, 4> Default DEFAULT();
};
