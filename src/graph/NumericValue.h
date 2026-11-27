#pragma once

#include "mesh/NamedAttribute.h"

namespace graph {
// The data types a socket carries.
// A color is three or four floats, and a matrix has no attribute storage.
// An image is a grid of values that image nodes sample.
enum class DataType : uint8_t { Float,
                                Vec2,
                                Vec3,
                                Vec4,
                                Color3,
                                Color4,
                                Matrix33,
                                Matrix44,
                                Int,
                                Bool,
                                Geometry,
                                Instances,
                                Image,
                                Reference };
// A value of a numeric data type, in the words of its components.
struct NumericValue {
    AttributeValue Words{};
    DataType Type{DataType::Float};
    bool operator==(const NumericValue &) const = default;
};
// The value of one port of a subgraph, by the port's id.
struct PortValue {
    uint16_t PortId{};
    NumericValue Value{};
    bool operator==(const PortValue &) const = default;
};
} // namespace graph
