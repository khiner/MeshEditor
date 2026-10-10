#pragma once

#include "action/Mesh.h"
#include "action/Updatable.h"
#include "gpu/AttributeNode.h"
#include "gpu/Transform.h"
#include "mesh/GeometrySelection.h"
#include "mesh/NamedAttribute.h"
#include "mesh/PrimitiveType.h"
#include "numeric/vec2.h"
#include "state/Entity.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <span>
#include <vector>

// A mesh entity's geometry procedure: node entities, each holding one parameter component, with their sockets linked output to input.
namespace procedure {
// Emits the mesh entity's source record.
struct Source {};
// Holds the procedure's result in its geometry input.
struct Output {};
// Appends its second geometry input's elements to its first's.
struct Join {};
// Combines its first geometry input's closed surface with its second's into a new mesh.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
};
// A primitive shape's parameters make a generator, which emits a new mesh of that shape.
using Generators = action::TypeList<primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;
template<typename K>
concept Generator = action::InTypeList<K, Generators>;

// Attribute nodes evaluate on the record arriving at their consumer's geometry input, in the domain of their first attribute input.
// Vertex positions.
struct Position {};
// Vertex or face normals.
struct Normal {
    AttributeDomain Domain{AttributeDomain::Vertex};
};
// Each element's rank among the live elements of its domain.
struct Index {
    AttributeDomain Domain{AttributeDomain::Vertex};
};
// The record's attribute of this name and type.
struct NamedAttribute {
    AttributeType Type{AttributeType::Float};
    AttributeName Name{};
};
// A vector operation on A and B, where B is the constant B, or Scale for Scale, when its input is unlinked.
struct VectorMath {
    VectorMathOp Operation{VectorMathOp::Add};
    vec3 B{};
    float Scale{1.f};
};
// A float operation on A and B, where B is the constant B when its input is unlinked.
struct FloatMath {
    FloatMathOp Operation{FloatMathOp::Add};
    float B{};
};
// Compares float values, or one component of vector values, where B is the constant B when its input is unlinked.
struct Compare {
    CompareOp Operation{CompareOp::Greater};
    float B{}, Epsilon{1e-5f};
    uint32_t Component{};
};
// A boolean operation on A and B, where B is the constant B when its input is unlinked.
struct BooleanMath {
    BooleanMathOp Operation{BooleanMathOp::And};
    bool B{};
};
// Stores its attribute input on its geometry under the name, replacing an attribute of that name.
struct StoreNamedAttribute {
    AttributeName Name{};
};
// Moves the selected vertices to a vec3 vertex attribute, or by it as an offset.
struct SetPosition {
    bool Offset{true};
};
// Moves the selected vertices by Delta about Pivot, both in the space Frame maps the record's positions into.
struct TransformElements {
    Transform Delta{};
    vec3 Pivot{};
    Transform Frame{};
};

// A node's kind is the index of its parameter component here.
using NodeKinds = action::TypeList<
    Source, Output, Join, Boolean,
    primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone,
    action::mesh::Delete, action::mesh::Merge, action::mesh::Extrude, action::mesh::Duplicate, action::mesh::Split, action::mesh::Dissolve,
    action::mesh::Subdivide, action::mesh::Triangulate, action::mesh::TrisToQuads, action::mesh::Poke, action::mesh::FlipNormals, action::mesh::EdgeSplit,
    action::mesh::Inset, action::mesh::Solidify, action::mesh::Wireframe, action::mesh::Bevel, action::mesh::SmoothVertices, action::mesh::ShrinkFatten,
    action::mesh::Randomize, action::mesh::Spin, action::mesh::ExtrudeRepeat, action::mesh::Bisect, action::mesh::Symmetrize, action::mesh::FillHoles,
    action::mesh::ConvexHull, action::mesh::Decimate, action::mesh::RecalculateNormals, action::mesh::Rip, TransformElements,
    Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, SetPosition>;

// An attribute socket carries an attribute of the record at its consumer's geometry input.
// A selection socket takes a bool attribute, and its node's inline selection while unlinked.
enum class SocketType : uint8_t { Geometry,
                                  Selection,
                                  Attribute };
enum class SocketDirection : uint8_t { In,
                                       Out };
constexpr uint8_t TypeBit(AttributeType type) { return uint8_t(1u << uint32_t(type)); }
inline constexpr uint8_t AnyType = (1u << AttributeTypeCount) - 1u;
struct Socket {
    SocketType Type;
    SocketDirection Direction;
    uint8_t Types{}; // The attribute types a non-geometry socket carries, one TypeBit each
};

namespace detail {
using enum SocketType;
using enum SocketDirection;
inline constexpr uint8_t Bool = TypeBit(AttributeType::Bool), Float = TypeBit(AttributeType::Float), Vec3 = TypeBit(AttributeType::Vec3);
// A source or generator emits geometry from no input.
inline constexpr Socket SourceSockets[]{{Geometry, Out}};
inline constexpr Socket OutputSockets[]{{Geometry, In}};
// A combining node reads two geometry inputs.
inline constexpr Socket CombineSockets[]{{Geometry, In}, {Geometry, In}, {Geometry, Out}};
// An operator edits its geometry input where its selection input selects.
inline constexpr Socket OperatorSockets[]{{Geometry, In}, {Selection, In, Bool}, {Geometry, Out}};
inline constexpr Socket PositionSockets[]{{Attribute, Out, Vec3}};
inline constexpr Socket IndexSockets[]{{Attribute, Out, TypeBit(AttributeType::Int)}};
inline constexpr Socket NamedAttributeSockets[]{{Attribute, Out, AnyType}};
inline constexpr Socket VectorMathSockets[]{{Attribute, In, Vec3}, {Attribute, In, Vec3 | Float}, {Attribute, Out, Vec3 | Float}};
inline constexpr Socket FloatMathSockets[]{{Attribute, In, Float}, {Attribute, In, Float}, {Attribute, Out, Float}};
inline constexpr Socket CompareSockets[]{{Attribute, In, Float | Vec3}, {Attribute, In, Float | Vec3}, {Attribute, Out, Bool}};
inline constexpr Socket BooleanMathSockets[]{{Attribute, In, Bool}, {Attribute, In, Bool}, {Attribute, Out, Bool}};
inline constexpr Socket StoreSockets[]{{Geometry, In}, {Attribute, In, AnyType}, {Geometry, Out}};
inline constexpr Socket SetPositionSockets[]{{Geometry, In}, {Attribute, In, Vec3}, {Selection, In, Bool}, {Geometry, Out}};
template<typename K> constexpr std::span<const Socket> SocketsOf() {
    if constexpr (std::same_as<K, Source> || Generator<K>) return SourceSockets;
    else if constexpr (std::same_as<K, Output>) return OutputSockets;
    else if constexpr (std::same_as<K, Join> || std::same_as<K, Boolean>) return CombineSockets;
    else if constexpr (std::same_as<K, Position> || std::same_as<K, Normal>) return PositionSockets;
    else if constexpr (std::same_as<K, Index>) return IndexSockets;
    else if constexpr (std::same_as<K, NamedAttribute>) return NamedAttributeSockets;
    else if constexpr (std::same_as<K, VectorMath>) return VectorMathSockets;
    else if constexpr (std::same_as<K, FloatMath>) return FloatMathSockets;
    else if constexpr (std::same_as<K, Compare>) return CompareSockets;
    else if constexpr (std::same_as<K, BooleanMath>) return BooleanMathSockets;
    else if constexpr (std::same_as<K, StoreNamedAttribute>) return StoreSockets;
    else if constexpr (std::same_as<K, SetPosition>) return SetPositionSockets;
    else return OperatorSockets;
}
template<typename... Ks> constexpr auto PerKind(action::TypeList<Ks...>, auto &&value) {
    return std::array<uint16_t, sizeof...(Ks)>{uint16_t(value.template operator()<Ks>())...};
}
} // namespace detail

inline constexpr auto SocketCount = detail::PerKind(NodeKinds{}, []<typename K> { return detail::SocketsOf<K>().size(); });
inline constexpr auto InputCount = detail::PerKind(NodeKinds{}, []<typename K> {
    return std::ranges::count(detail::SocketsOf<K>(), SocketDirection::In, &Socket::Direction);
});
inline constexpr uint16_t KindCount = uint16_t(SocketCount.size());
// Each kind's sockets, inputs before outputs, start at SocketFirst[kind] of Sockets.
inline constexpr auto SocketFirst = [] {
    std::array<uint16_t, KindCount + 1u> first{};
    for (uint16_t k = 0u; k < KindCount; ++k) first[k + 1u] = first[k] + SocketCount[k];
    return first;
}();
inline constexpr auto Sockets = []<typename... Ks>(action::TypeList<Ks...>) {
    std::array<Socket, SocketFirst.back()> all{};
    auto it = all.begin();
    ((it = std::ranges::copy(detail::SocketsOf<Ks>(), it).out), ...);
    return all;
}(NodeKinds{});
static_assert([] {
    for (uint16_t k = 0u; k < KindCount; ++k)
        for (uint16_t s = 0u; s < SocketCount[k]; ++s)
            if ((s < InputCount[k]) != (Sockets[SocketFirst[k] + s].Direction == SocketDirection::In)) return false;
    return true;
}(),
              "A kind's inputs precede its outputs");

constexpr std::span<const Socket> KindSockets(uint16_t kind) { return std::span{Sockets}.subspan(SocketFirst[kind], SocketCount[kind]); }
// An attribute node has no geometry socket, and evaluates once for each node that reads it on that node's geometry input.
constexpr bool IsAttributeKind(uint16_t kind) { return KindSockets(kind).front().Type == SocketType::Attribute; }

template<typename K> inline constexpr uint16_t KindOf = []<typename... Ks>(action::TypeList<Ks...>) {
    uint16_t index = 0u, found = KindCount;
    ((found = std::same_as<K, Ks> ? index : found, ++index), ...);
    return found;
}(NodeKinds{});

// Invokes `f.template operator()<K>()` for the kind's parameter component.
template<typename F> void VisitKind(uint16_t kind, F &&f) {
    [&]<typename... Ks>(action::TypeList<Ks...>) {
        uint16_t index = 0u;
        (void)((index++ == kind && (f.template operator()<Ks>(), true)) || ...);
    }(NodeKinds{});
}
template<typename F> void ForEachKind(F &&f) {
    [&]<typename... Ks>(action::TypeList<Ks...>) { (f.template operator()<Ks>(), ...); }(NodeKinds{});
}

inline constexpr uint32_t NoNode = UINT32_MAX;
struct NodeRecord {
    state::Entity Entity;
    uint16_t Kind;
};
// The output socket an input reads, with NoNode for an unlinked input.
struct SocketLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
};
// Persistent on the mesh entity.
struct GeometryProcedure {
    std::vector<NodeRecord> Nodes; // In authoring order
    std::vector<SocketLink> Inputs; // Each node's input sockets, in node order
    uint32_t OutputNode{NoNode}; // The Output node whose input holds the result
};
// The node's first input in Inputs.
inline uint32_t InputFirst(const GeometryProcedure &procedure, uint32_t node) {
    uint32_t first = 0u;
    for (uint32_t n = 0u; n < node; ++n) first += InputCount[procedure.Nodes[n].Kind];
    return first;
}
// Each node's first input in Inputs, then the input count.
inline std::vector<uint32_t> InputFirsts(const GeometryProcedure &procedure) {
    std::vector<uint32_t> first{0u};
    for (const auto &node : procedure.Nodes) first.push_back(first.back() + InputCount[node.Kind]);
    return first;
}
// Marks `node` and the nodes it reads through its inputs, directly or through other marked nodes, where `follows` admits each node it marks.
// Each node's inputs start at its entry of `input_first`.
std::vector<bool> Upstream(const GeometryProcedure &procedure, std::span<const uint32_t> input_first, uint32_t node, auto &&follows) {
    std::vector<bool> marked(procedure.Nodes.size());
    marked[node] = true;
    for (std::vector<uint32_t> stack{node}; !stack.empty();) {
        const auto n = stack.back();
        stack.pop_back();
        for (uint32_t i = input_first[n]; i < input_first[n + 1u]; ++i)
            if (const auto from = procedure.Inputs[i].FromNode; from != NoNode && !marked[from] && follows(from)) {
                marked[from] = true;
                stack.push_back(from);
            }
    }
    return marked;
}

// Persistent on a node, for the node editor.
struct NodeLayout {
    vec2 Position, Size;
};
// Persistent on a node: its selection input as one mask block per block its input record owns, in ascending block order.
// An absent or empty selection selects every element.
struct InlineSelection {
    std::vector<SelectionBlock> Vertices, Edges, Faces;
};
// Persistent on a node a viewport operator appended, whose edit selects the elements it emits in its output record the way the operator does.
struct PublishSelection {};
// Derived on a node whose input is missing or whose operation changed nothing, which forwards its input.
struct NodeError {};
// Derived on the mesh entity, rebuilt with its procedure.
struct CookPlan {
    std::vector<uint32_t> Order; // Every node after the nodes it reads
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input and first socket slot, then the totals
    std::vector<uint32_t> Consumers; // The inputs reading each socket slot
};
} // namespace procedure
