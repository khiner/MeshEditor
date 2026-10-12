#pragma once

#include "gpu/NamedAttributeRef.h"
#include "gpu/SlotOffset.h"

enum class VectorMathOp : uint32_t {
    Add,
    Subtract,
    Multiply,
    Scale,
    Dot,
    Cross,
    Length,
    Normalize,
    Distance,
};
enum class FloatMathOp : uint32_t {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
    Abs,
    Power,
    Sqrt,
    Sin,
    Cos,
};
enum class CompareOp : uint32_t {
    Less,
    Greater,
    Equal,
};
enum class BooleanMathOp : uint32_t {
    And,
    Or,
    Not,
    Xor,
};

// Where an attribute node input reads each element's value.
enum class AttributeSource : uint32_t {
    Constant, // Value at every element
    Named, // Attribute's value at the element
    Positions, // Vertex position in the vertex arena at Slot
    Vectors, // Packed float3 at the element's arena address in the buffer at Slot
    Ordinal, // Element's rank among the record's live elements of the domain
};
struct AttributeInput {
    AttributeSource Source DEFAULT(AttributeSource::Constant);
    AttributeType Type DEFAULT(AttributeType::Float);
    NamedAttributeRef Attribute DEFAULT();
    uint32_t Slot DEFAULT(InvalidSlot);
    GpuArray<uint32_t, 4> Value DEFAULT(); // The constant's component words
};
static_assert(sizeof(AttributeInput) == 36);

// One attribute node evaluated at every live element of a record's domain, one thread per element slot of its blocks.
struct AttributeNodePushConstants {
    SlotOffset Blocks; // The record's ascending domain blocks, followed by their inclusive live prefix
    uint32_t BlockCount, MembershipSlot;
    AttributeInput A, B;
    NamedAttributeRef Output;
    AttributeType OutputType;
    uint32_t Operation, Component;
    float Epsilon;
};
static_assert(sizeof(AttributeNodePushConstants) == 112);

// Copies a bool attribute's words at each element block of its record to the block's mask block, and zero to the mask blocks of other blocks.
// Mask block b, starting at word Masks.Offset + 8b, holds element block b, and an attribute without a slot names a domain it is not on.
struct AttributeMaskPushConstants {
    NamedAttributeRef Attribute;
    uint32_t MembershipSlot, Owner; // The domain's block metadata and the record's set in it
    SlotOffset Masks;
    uint32_t Words;
};
static_assert(sizeof(AttributeMaskPushConstants) == 28);
