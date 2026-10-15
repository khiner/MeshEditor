#pragma once

#include "Field.h"
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
#include <string_view>
#include <utility>
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
enum class VectorComponent : uint8_t { X,
                                       Y,
                                       Z };
// Compares float values, or one component of vector values, where B is the constant B when its input is unlinked.
struct Compare {
    CompareOp Operation{CompareOp::Greater};
    float B{}, Epsilon{1e-5f};
    VectorComponent Component{VectorComponent::X};
};
// A boolean operation on A and B, where B is the constant B when its input is unlinked.
struct BooleanMath {
    BooleanMathOp Operation{BooleanMathOp::And};
    bool B{};
};
// Stores its value input on its geometry under the name, replacing an attribute of that name.
// An attribute keeps its domain, and a single value fills an attribute of Domain.
struct StoreNamedAttribute {
    AttributeName Name{};
    AttributeDomain Domain{AttributeDomain::Vertex};
};
// A value node emits one value, which a single-value parameter socket takes, and an attribute socket takes at every element.
struct FloatValue {
    float Value{};
};
struct IntegerValue {
    uint32_t Value{};
};
struct BooleanValue {
    bool Value{};
};
struct VectorValue {
    vec3 Value{};
};
struct RotationValue {
    quat Value{1, 0, 0, 0};
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
// Moves every vertex by Delta about the origin, as Blender's Transform Geometry does.
struct TransformGeometry {
    Transform Delta{};
};

// An instances value is a face-less record whose vertices are its points, each placing one instance of its instance geometry.
// The point is the instance's position, and these vertex attributes hold its rotation as a quaternion and its scale.
inline constexpr std::string_view InstanceRotation = ".instance.rotation", InstanceScale = ".instance.scale";
// Emits its geometry's vertices as points.
struct MeshToPoints {};
// Places an instance of its second geometry input at each vertex of its first, rotated by Rotation and scaled by Scale, each per point through its attribute socket.
struct InstanceOnPoints {
    vec3 Scale{1, 1, 1};
    quat Rotation{1, 0, 0, 0};
};
// Emits one copy of the instance geometry per instance, moved by the instance's transform.
struct RealizeInstances {};
// Emits the drawn record of the object's mesh in its local space, which the cook reads in place.
// An object without a mesh, or whose mesh the procedure is on, emits nothing.
struct ObjectInfo {
    state::Entity Object{state::Null};
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
    Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, SetPosition,
    MeshToPoints, InstanceOnPoints, RealizeInstances, ObjectInfo,
    FloatValue, IntegerValue, BooleanValue, VectorValue, TransformGeometry, RotationValue>;
using ValueKinds = action::TypeList<FloatValue, IntegerValue, BooleanValue, VectorValue, RotationValue>;
template<typename K>
concept ValueKind = action::InTypeList<K, ValueKinds>;

// An attribute socket carries an attribute of the record at its consumer's geometry input, or one value at every element.
// A value socket carries one value, which a single-value parameter takes.
// A selection socket takes a bool attribute, a bool value, and its node's inline selection while unlinked.
// An instances socket carries points that each place an instance, which an Output node's geometry input also takes.
enum class SocketType : uint8_t { Geometry,
                                  Selection,
                                  Attribute,
                                  Instances,
                                  Value };
enum class SocketDirection : uint8_t { In,
                                       Out };
constexpr uint8_t TypeBit(AttributeType type) { return uint8_t(1u << uint32_t(type)); }
inline constexpr uint8_t AnyType = (1u << AttributeTypeCount) - 1u;
// A parameter socket carries one field of its node's parameters by the field's declaration index, or one member of a nested field by both indices.
inline constexpr uint8_t NoField = UINT8_MAX;
inline constexpr size_t MaxParameterFields = 16u;
struct Socket {
    SocketType Type;
    SocketDirection Direction;
    uint8_t Types{}; // The attribute types a non-geometry socket carries, one TypeBit each
    bool Reads{}; // A geometry input the node only reads, where every other geometry input edits its record
    const char *Name{}; // The editor's label, or the type's name when null
    uint8_t Field{NoField};
    uint8_t Member{NoField};
    // Whether the socket carries the parameter field at the path.
    constexpr bool Carries(uint8_t field, uint8_t member = NoField) const { return Field == field && Member == member; }
};

namespace detail {
using enum SocketType;
using enum SocketDirection;
inline constexpr uint8_t Bool = TypeBit(AttributeType::Bool), Float = TypeBit(AttributeType::Float), Vec3 = TypeBit(AttributeType::Vec3);
// A kind's fixed sockets, which its parameter sockets follow after the inputs.
// A source or generator emits geometry from no input.
inline constexpr Socket SourceSockets[]{{Geometry, Out}};
inline constexpr Socket OutputSockets[]{{Geometry, In}};
// A join moves the elements of its second input into the geometry it edits.
inline constexpr Socket JoinSockets[]{{Geometry, In}, {Geometry, In}, {Geometry, Out}};
inline constexpr Socket BooleanSockets[]{{Geometry, In, 0u, true, "Mesh 1"}, {Geometry, In, 0u, true, "Mesh 2"}, {Geometry, Out}};
// An operator edits its geometry input where its selection input selects.
inline constexpr Socket OperatorSockets[]{{Geometry, In}, {Selection, In, Bool}, {Geometry, Out}};
inline constexpr Socket PositionSockets[]{{Attribute, Out, Vec3}};
inline constexpr Socket IndexSockets[]{{Attribute, Out, TypeBit(AttributeType::Int)}};
inline constexpr Socket NamedAttributeSockets[]{{Attribute, Out, AnyType}};
inline constexpr Socket VectorMathSockets[]{{Attribute, In, Vec3, false, "A"}, {Attribute, Out, Vec3 | Float, false, "Result"}};
inline constexpr Socket FloatMathSockets[]{{Attribute, In, Float, false, "A"}, {Attribute, Out, Float, false, "Result"}};
inline constexpr Socket CompareSockets[]{{Attribute, In, Float | Vec3, false, "A"}, {Attribute, Out, Bool, false, "Result"}};
inline constexpr Socket BooleanMathSockets[]{{Attribute, In, Bool, false, "A"}, {Attribute, Out, Bool, false, "Result"}};
inline constexpr Socket StoreSockets[]{{Geometry, In}, {Attribute, In, AnyType, false, "Value"}, {Geometry, Out}};
inline constexpr Socket SetPositionSockets[]{{Geometry, In}, {Attribute, In, Vec3, false, "Position"}, {Selection, In, Bool}, {Geometry, Out}};
inline constexpr Socket PointsSockets[]{{Geometry, In}, {Geometry, Out, 0u, false, "Points"}};
// A whole-geometry operator edits every element of its input.
inline constexpr Socket WholeSockets[]{{Geometry, In}, {Geometry, Out}};
// Instancing edits its points and reads its instance geometry.
inline constexpr Socket InstanceSockets[]{{Geometry, In, 0u, false, "Points"}, {Geometry, In, 0u, true, "Instance"}, {Instances, Out}};
inline constexpr Socket RealizeSockets[]{{Instances, In, 0u, true}, {Geometry, Out}};
// The attribute type a parameter field of type F carries through its socket, or none for a field without a socket.
template<typename F> constexpr uint8_t TypesOf() {
    if constexpr (std::same_as<F, float>) return Float;
    else if constexpr (std::same_as<F, uint32_t>) return TypeBit(AttributeType::Int);
    else if constexpr (std::same_as<F, bool>) return Bool;
    else if constexpr (std::same_as<F, vec3>) return Vec3;
    else if constexpr (std::same_as<F, quat>) return TypeBit(AttributeType::Vec4);
    else return 0u;
}
// A nested field's socket is named by its field's label and its member's, where a type names its members as its editor does.
template<typename T, size_t J> inline constexpr auto MemberLabel = field::Label<T, J>;
template<> inline constexpr auto MemberLabel<Transform, 0> = field::fixed_string{"Position"};
template<> inline constexpr auto MemberLabel<Transform, 1> = field::fixed_string{"Rotation"};
template<> inline constexpr auto MemberLabel<Transform, 2> = field::fixed_string{"Scale"};
template<size_t A, size_t B> consteval auto Spaced(field::fixed_string<A> a, field::fixed_string<B> b) {
    field::fixed_string<A + B> out{};
    for (size_t i = 0u; i + 1u < A; ++i) out.Chars[i] = a.Chars[i];
    out.Chars[A - 1u] = ' ';
    for (size_t i = 0u; i + 1u < B; ++i) out.Chars[A + i] = b.Chars[i];
    return out;
}
template<typename K, size_t I, typename F, size_t J> inline constexpr auto NestedLabel = Spaced(field::Label<K, I>, MemberLabel<F, J>);
template<typename K> inline constexpr Socket ValueSockets[]{{Value, Out, TypesOf<decltype(K::Value)>()}};
template<typename K> constexpr std::span<const Socket> FixedSockets() {
    if constexpr (std::same_as<K, Source> || std::same_as<K, ObjectInfo> || Generator<K>) return SourceSockets;
    else if constexpr (std::same_as<K, Output>) return OutputSockets;
    else if constexpr (std::same_as<K, Join>) return JoinSockets;
    else if constexpr (std::same_as<K, Boolean>) return BooleanSockets;
    else if constexpr (std::same_as<K, Position> || std::same_as<K, Normal>) return PositionSockets;
    else if constexpr (std::same_as<K, Index>) return IndexSockets;
    else if constexpr (std::same_as<K, NamedAttribute>) return NamedAttributeSockets;
    else if constexpr (std::same_as<K, VectorMath>) return VectorMathSockets;
    else if constexpr (std::same_as<K, FloatMath>) return FloatMathSockets;
    else if constexpr (std::same_as<K, Compare>) return CompareSockets;
    else if constexpr (std::same_as<K, BooleanMath>) return BooleanMathSockets;
    else if constexpr (std::same_as<K, StoreNamedAttribute>) return StoreSockets;
    else if constexpr (std::same_as<K, SetPosition>) return SetPositionSockets;
    else if constexpr (std::same_as<K, MeshToPoints>) return PointsSockets;
    else if constexpr (std::same_as<K, TransformGeometry>) return WholeSockets;
    else if constexpr (std::same_as<K, InstanceOnPoints>) return InstanceSockets;
    else if constexpr (std::same_as<K, RealizeInstances>) return RealizeSockets;
    else if constexpr (ValueKind<K>) return ValueSockets<K>;
    else return OperatorSockets;
}

// The attribute types a kind's parameter field carries through an attribute socket, one value per element of the domain the node edits it in, declared per field the way its Spec is.
// Every other parameter field of a float, integer, bool or vec3 type has a value socket.
template<typename K, field::fixed_string Name> inline constexpr uint8_t PerElement = 0u;
template<> inline constexpr uint8_t PerElement<action::mesh::Bevel, "Width"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Poke, "Offset"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Solidify, "Thickness"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Wireframe, "Thickness"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Inset, "Thickness"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Inset, "Depth"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::ShrinkFatten, "Distance"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::SmoothVertices, "Factor"> = Float;
template<> inline constexpr uint8_t PerElement<action::mesh::Randomize, "Amount"> = Float;
template<> inline constexpr uint8_t PerElement<InstanceOnPoints, "Scale"> = Vec3;
template<> inline constexpr uint8_t PerElement<InstanceOnPoints, "Rotation"> = TypeBit(AttributeType::Vec4);
template<> inline constexpr uint8_t PerElement<VectorMath, "B"> = Vec3;
template<> inline constexpr uint8_t PerElement<VectorMath, "Scale"> = Float;
template<> inline constexpr uint8_t PerElement<FloatMath, "B"> = Float;
template<> inline constexpr uint8_t PerElement<Compare, "B"> = Float | Vec3;
template<> inline constexpr uint8_t PerElement<BooleanMath, "B"> = Bool;
// The kind's sockets with their count: its fixed inputs, an input per parameter field of a socket type in field order, then its fixed outputs.
// A value kind's one field is its output.
template<typename K> inline constexpr auto KindSocketTable = [] {
    constexpr auto fixed = FixedSockets<K>();
    std::array<Socket, fixed.size() + MaxParameterFields> all{};
    size_t n = 0u;
    for (const auto &socket : fixed) {
        if (socket.Direction == In) all[n++] = socket;
    }
    if constexpr (field::Walkable<K> && !ValueKind<K>) {
        K value{};
        field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
            using F = std::remove_cvref_t<decltype(member)>;
            static_assert(I < MaxParameterFields);
            constexpr auto types = TypesOf<F>();
            if constexpr (types != 0u) {
                constexpr auto per_element = PerElement<K, field::NameString<K, I>>;
                all[n++] = {per_element ? Attribute : Value, In, per_element ? per_element : types, false, field::Label<K, I>.c_str(), uint8_t(I)};
            } else if constexpr (field::Walkable<F>) {
                // A nested field's members each take a value socket.
                field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                    constexpr auto nested_types = TypesOf<std::remove_cvref_t<decltype(nested)>>();
                    if constexpr (nested_types != 0u) {
                        static_assert(J < MaxParameterFields);
                        all[n++] = {Value, In, nested_types, false, NestedLabel<K, I, F, J>.c_str(), uint8_t(I), uint8_t(J)};
                    }
                });
            }
        });
    }
    for (const auto &socket : fixed) {
        if (socket.Direction == Out) all[n++] = socket;
    }
    return std::pair{all, n};
}();
template<typename K> constexpr std::span<const Socket> SocketsOf() { return std::span{KindSocketTable<K>.first}.first(KindSocketTable<K>.second); }
template<typename... Ks> constexpr auto PerKind(action::TypeList<Ks...>, auto &&value) {
    return std::array<uint16_t, sizeof...(Ks)>{uint16_t(value.template operator()<Ks>())...};
}
} // namespace detail

inline constexpr auto SocketCount = detail::PerKind(NodeKinds{}, []<typename K> { return detail::SocketsOf<K>().size(); });
inline constexpr auto InputCount = detail::PerKind(NodeKinds{}, []<typename K> {
    return std::ranges::count(detail::SocketsOf<K>(), SocketDirection::In, &Socket::Direction);
});
inline constexpr uint16_t KindCount = uint16_t(SocketCount.size());
inline constexpr size_t MaxInputs = std::ranges::max(InputCount);
// Each kind's sockets, inputs before outputs, start at SocketFirst[kind] of Sockets.
inline constexpr auto SocketFirst = [] {
    std::array<uint16_t, KindCount + 1u> first{};
    for (uint16_t k = 0u; k < KindCount; ++k) first[k + 1u] = first[k] + SocketCount[k];
    return first;
}();
inline constexpr auto Sockets = []<typename... Ks>(action::TypeList<Ks...>) {
    std::array<Socket, SocketFirst.back()> all{};
    auto *it = all.begin();
    ((it = std::ranges::copy(detail::SocketsOf<Ks>(), it).out), ...);
    return all;
}(NodeKinds{});
static_assert([] {
    for (uint16_t k = 0u; k < KindCount; ++k) {
        for (uint16_t s = 0u; s < SocketCount[k]; ++s) {
            if ((s < InputCount[k]) != (Sockets[SocketFirst[k] + s].Direction == SocketDirection::In)) return false;
        }
    }
    return true;
}(),
              "A kind's inputs precede its outputs");

constexpr std::span<const Socket> KindSockets(uint16_t kind) { return std::span{Sockets}.subspan(SocketFirst[kind], SocketCount[kind]); }
// An attribute node has no geometry socket, and evaluates once for each node that reads it on that node's geometry input.
constexpr bool IsAttributeKind(uint16_t kind) { return KindSockets(kind).front().Type == SocketType::Attribute; }
// The range of the kind's parameter sockets, which follow its fixed inputs and end its inputs.
constexpr std::pair<uint16_t, uint16_t> ParameterRange(uint16_t kind) {
    return {uint16_t(std::ranges::count(KindSockets(kind).first(InputCount[kind]), NoField, &Socket::Field)), InputCount[kind]};
}

template<typename K> inline constexpr uint16_t KindOf = []<typename... Ks>(action::TypeList<Ks...>) {
    uint16_t index = 0u, found = KindCount;
    ((found = std::same_as<K, Ks> ? index : found, ++index), ...);
    return found;
}(NodeKinds{});

// Whether an output socket links into an input socket of a node of `to_kind`: the same type sharing an attribute type, a bool attribute into a selection, or a value into an attribute or selection input sharing its type.
// An Output node's geometry input also takes instances, which its mesh entity publishes.
constexpr bool Links(Socket from, Socket to, uint16_t to_kind) {
    if (from.Type == SocketType::Instances) return to.Type == SocketType::Instances || to_kind == KindOf<Output>;
    const bool typed = from.Type == to.Type || ((from.Type == SocketType::Attribute || from.Type == SocketType::Value) && to.Type == SocketType::Selection) || (from.Type == SocketType::Value && to.Type == SocketType::Attribute);
    return typed && (from.Type == SocketType::Geometry || (from.Types & to.Types));
}

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
    uint32_t ActiveNode{NoNode}; // The node the editor's parameter panel edits
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
        for (uint32_t i = input_first[n]; i < input_first[n + 1u]; ++i) {
            if (const auto from = procedure.Inputs[i].FromNode; from != NoNode && !marked[from] && follows(from)) {
                marked[from] = true;
                stack.push_back(from);
            }
        }
    }
    return marked;
}
// The kind's first geometry socket in the direction, through which a node inserted on a link links, or none.
inline std::optional<uint16_t> MainSocket(uint16_t kind, SocketDirection direction) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Type == SocketType::Geometry && sockets[s].Direction == direction) return s;
    }
    return {};
}
// Whether the node inserts on the link between two other nodes that enters the input: its main output links into the input, the link's source links into its main input, and neither link closes a cycle.
inline bool Inserts(const GeometryProcedure &procedure, std::span<const uint32_t> input_first, uint32_t node, uint32_t to_node, uint16_t to_socket) {
    const auto link = procedure.Inputs[input_first[to_node] + to_socket];
    const auto kind = procedure.Nodes[node].Kind, to_kind = procedure.Nodes[to_node].Kind;
    const auto input = MainSocket(kind, SocketDirection::In), output = MainSocket(kind, SocketDirection::Out);
    if (link.FromNode == NoNode || link.FromNode == node || to_node == node || !input || !output) return false;
    const auto from_kind = procedure.Nodes[link.FromNode].Kind;
    if (!Links(KindSockets(kind)[*output], KindSockets(to_kind)[to_socket], to_kind) || !Links(KindSockets(from_kind)[link.FromSocket], KindSockets(kind)[*input], kind)) return false;
    const auto all = [](uint32_t) { return true; };
    return !Upstream(procedure, input_first, node, all)[to_node] && !Upstream(procedure, input_first, link.FromNode, all)[node];
}
// Persistent on a mesh entity whose procedure results in instances: the instance entities it publishes, one per point under each object instancing the mesh, object by object.
// Each entity is an Instance of the instanced mesh whose Transform is its point's instance transform.
struct ProcedureInstances {
    std::vector<state::Entity> Entities;
};
// Persistent on a mesh entity whose procedure results in instances of geometry the procedure made: the mesh entity holding that geometry, which the published instances instance.
// The owned mesh entity has no object, transform or name of its own, and it destroys once the result stops instancing it and with the procedure.
struct ProcedureOwnedMesh {
    state::Entity MeshEntity;
};

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
// Persistent on a node the editor has selected, within its procedure's own selection.
struct NodeSelected {};
// Why a node could not run, which forwards its input instead.
// An operation that runs and changes nothing, or whose geometry input is unlinked, forwards its input without a fault.
enum class NodeFault : uint8_t {
    Selection, // The selection does not fit the operation
    Parameter, // A parameter is invalid
    Attribute, // An attribute input is missing or has the wrong type or domain
    NotClosed, // A boolean input is not a closed surface
    Empty, // A boolean produced no geometry
    NoMesh, // An Object Info node names no mesh it can read
};
// Derived on a node that could not run.
struct NodeError {
    NodeFault Fault;
};
// Derived on the mesh entity, rebuilt with its procedure.
struct CookPlan {
    std::vector<uint32_t> Order; // Every node after the nodes it reads
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input and first socket slot, then the totals
    std::vector<uint32_t> Consumers; // The inputs reading each socket slot
};
} // namespace procedure