#pragma once

#include "gpu/ElementWork.h"
#include "gpu/NamedAttributeRef.h"

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
// The operations of Blender's Math node, in its menu order.
enum class FloatMathOp : uint32_t {
    // Functions
    Add,
    Subtract,
    Multiply,
    Divide,
    MultiplyAdd,
    Power,
    Logarithm,
    SquareRoot,
    InverseSquareRoot,
    Absolute,
    Exponent,
    // Comparison
    Minimum,
    Maximum,
    LessThan,
    GreaterThan,
    Sign,
    Compare,
    SmoothMinimum,
    SmoothMaximum,
    // Rounding
    Round,
    Floor,
    Ceil,
    Truncate,
    Fraction,
    TruncatedModulo,
    FlooredModulo,
    Wrap,
    Snap,
    PingPong,
    // Trigonometric
    Sine,
    Cosine,
    Tangent,
    Arcsine,
    Arccosine,
    Arctangent,
    Arctan2,
    HyperbolicSine,
    HyperbolicCosine,
    HyperbolicTangent,
    // Conversion
    ToRadians,
    ToDegrees,
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

enum class AttributeSource : uint32_t {
    Constant,
    Named,
    Positions, // Vertex position in the vertex arena at Slot
    Vectors, // Packed float3 at the element's arena address in the buffer at Slot
    Ordinal, // Element's rank among the record's live elements of the domain
};
struct AttributeInput {
    AttributeSource Source DEFAULT(AttributeSource::Constant);
    AttributeType Type DEFAULT(AttributeType::Float);
    NamedAttributeRef Attribute DEFAULT();
    uint32_t Slot DEFAULT(InvalidSlot);
    GpuArray<uint32_t, 4> Value DEFAULT();
};

// One attribute node evaluated at every live element of a record's domain, one thread per element slot of its blocks.
struct AttributeNodePushConstants {
    DomainBlocks Blocks;
    AttributeInput A, B, C;
    NamedAttributeRef Output;
    AttributeType OutputType;
    uint32_t Operation, Component;
    float Epsilon;
    uint32_t Clamp; // Nonzero clamps a float math result to [0, 1]
};

// Copies a bool attribute's words at each element block of its record to the block's mask block, and zero to the mask blocks of other blocks.
// Mask block b covers element block b at word Masks.Offset + 8b.
// An absent slot denotes an attribute absent from that domain.
struct AttributeMaskPushConstants {
    NamedAttributeRef Attribute;
    uint32_t MembershipSlot, Owner; // The domain's block metadata and the record's set in it
    SlotOffset Masks;
    uint32_t Words;
};
