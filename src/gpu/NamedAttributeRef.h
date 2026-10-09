#pragma once
#include "gpu/Types.h"

// Value types of named attributes, each stored in its own payload arena.
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

// A named attribute's payload blocks, found through its run of the shared table arena.
// Table word TableOffset + b - FirstBlock holds element block b's payload block plus one, and zero when the block has none.
struct NamedAttributeRef {
    uint32_t TablesSlot DEFAULT(InvalidSlot);
    uint32_t ValuesSlot DEFAULT(InvalidSlot);
    uint32_t TableOffset DEFAULT();
    uint32_t FirstBlock DEFAULT();
};
static_assert(sizeof(NamedAttributeRef) == 16);

// One named attribute a topology job carries from its source elements to its output elements, in its domain's run of the job's records.
// Default holds the value bits of output elements without a source.
struct TopologyAttribute {
    AttributeType Type DEFAULT();
    NamedAttributeRef Source DEFAULT(), Destination DEFAULT();
    GpuArray<uint32_t, 4> Default DEFAULT();
};
static_assert(sizeof(TopologyAttribute) == 52);
