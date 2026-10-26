#pragma once

#include "action/Mesh.h"
#include "action/Updatable.h"
#include "gpu/AttributeNode.h"
#include "mesh/NamedAttribute.h"
#include "procedure/ProcedureSchema.h"

// Each procedure node is an entity with one parameter component and input/output links.
namespace procedure {
struct Source {}; // Emit the mesh's source geometry.
struct Output {}; // Receive the procedure's result.
struct Join {}; // Append the second input's geometry to the first.

// Combine the inputs in the space Frame1 maps Mesh 1 into and Frame2 maps Mesh 2 into, and emit the result in Mesh 1's space.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
    Transform Frame1{}, Frame2{};
};

using Generators = TypeList<primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;
template<typename K>
concept Generator = InTypeList<K, Generators>;

// Attribute nodes evaluate on the geometry their reader receives, in the domain of their first attribute input.
struct Position {}; // Emit vertex positions.

struct Normal {
    AttributeDomain Domain{AttributeDomain::Vertex};
};

// Emit each element's index in its domain, counting live elements only.
struct Index {
    AttributeDomain Domain{AttributeDomain::Vertex};
};

struct NamedAttribute {
    AttributeType Type{AttributeType::Float};
    AttributeName Name{};
};

// Apply a vector operation to A and B.
// An unlinked B or Scale input uses its constant.
struct VectorMath {
    VectorMathOp Operation{VectorMathOp::Add};
    vec3 B{};
    float Scale{1.f};
};

struct FloatMath {
    FloatMathOp Operation{FloatMathOp::Add};
    float B{};
};

enum class VectorComponent : uint8_t { X,
                                       Y,
                                       Z };

struct Compare {
    CompareOp Operation{CompareOp::Greater};
    float B{}, Epsilon{1e-5f};
    VectorComponent Component{VectorComponent::X};
};

struct BooleanMath {
    BooleanMathOp Operation{BooleanMathOp::And};
    bool B{};
};

// Replace the named attribute, preserving attribute-input domains or filling Domain for a constant.
struct StoreNamedAttribute {
    AttributeName Name{};
    AttributeDomain Domain{AttributeDomain::Vertex};
};

// Replace the named Bool attribute with the selection, in a linked attribute's domain or else the edit domain of the inline selection.
struct StoreSelection {
    AttributeName Name{};
};

// Broadcast constant values received by attribute sockets.
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

// Set the selected vertices' positions from a vec3 attribute, or offset them by it.
struct SetPosition {
    bool Offset{true};
};

// Move the selected vertices by Delta around Pivot, in the space Frame maps positions into.
struct TransformElements {
    Transform Delta{};
    vec3 Pivot{};
    Transform Frame{};
};

struct TransformGeometry {
    Transform Delta{};
};

// These position operators apply in the space Frame maps mesh-local positions into.
// A viewport operator sets Frame to the world transform of its mesh's primary instance.

// Blend the selected vertices toward their mean distance from Center.
struct ToSphere {
    float Factor{1.f};
    vec3 Center{};
    Transform Frame{};
};
// Move the selected vertices toward Center by Distance, or away from it for a negative distance.
struct PushPull {
    float Distance{0.1f};
    vec3 Center{};
    Transform Frame{};
};
// Move the selected vertices along Direction by tan(Angle) times their offset from Center along Cross(Direction, Normal).
struct Shear {
    float Angle{0.2f};
    vec3 Center{}, Direction{1, 0, 0}, Normal{0, 0, 1};
    Transform Frame{};
};
// Warp around Center in Orientation.
// Compute automatic bounds from the selected vertices after OffsetAngle.
struct Warp {
    float Angle{2 * std::numbers::pi_v<float>}, OffsetAngle{0.f};
    vec3 Center{};
    bool AutoRange{true};
    float Min{-1.f}, Max{1.f};
    quat Orientation{};
    Transform Frame{};
};
// Bend along the rolled X axis of Orientation from Center over Radius units.
// Clamp fixes the start side and rigidly rotates the end side.
struct Bend {
    float Angle{0.78539816f}, Radius{1.f}, OffsetAngle{0.f};
    vec3 Center{};
    bool Clamp{true};
    quat Orientation{};
    Transform Frame{};
};
// Slide each selected vertex along its incident edge nearest Direction.
// Even uses the edge length of the active selected vertex, or else the first selected vertex.
struct VertexSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    Transform Frame{};
};
// Slide selected edge chains along adjacent faces toward the side Direction selects.
// Even uses the active selected vertex, or else the first selected vertex.
struct EdgeSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    Transform Frame{};
};
// Flatten face regions and remaining edge components independently.
// View flattens along ViewNormal, and X, Y and Z constrain local coordinates.
struct Flatten {
    action::mesh::FlattenMethod Method{action::mesh::FlattenMethod::LeastSquares};
    float Factor{1.f};
    vec3 ViewNormal{0, 0, 1};
    bool X{true}, Y{true}, Z{true};
    Transform Frame{};
};

// Set the sharpness of the selected elements by the edit domain their selection was captured in.
// Faces take Sharp as their flat shading, edges as their mark, and vertices mark every edge touching them.
struct MarkSharp {
    bool Sharp{true};
};

// Cut every edge whose projection crosses the segment from Start to End, in pixels of a view Extent pixels wide and high.
// ViewProjection maps Frame space into the view's clip space.
// Each cut divides its edge at the crossing's fraction along the projected edge.
struct Knife {
    vec2 Start{}, End{}, Extent{1, 1};
    mat4 ViewProjection{1.f};
    Transform Frame{};
};

// Shade every face smooth and clear every sharp mark, or shade every face flat.
struct ShadeSmooth {
    bool Smooth{true};
};
// Shade every face smooth and mark sharp each edge whose faces' normals differ by more than Angle.
struct ShadeSmoothByAngle {
    float Angle{std::numbers::pi_v<float> / 6.f};
};

// Instances use face-less point records with per-point rotation and scale attributes.
inline constexpr std::string_view InstanceRotation = ".instance.rotation", InstanceScale = ".instance.scale";

struct MeshToPoints {}; // Emit the input's vertices as points.

// Place an instance of the second input at each point of the first, with Rotation and Scale per point.
struct InstanceOnPoints {
    vec3 Scale{1, 1, 1};
    quat Rotation{1, 0, 0, 0};
};

struct RealizeInstances {}; // Emit a copy of the instance geometry per instance, moved by its transform.

// Borrow the object mesh in local space.
// Emit nothing for a missing mesh or this procedure mesh.
struct ObjectInfo {
    state::Entity Object{state::Null};
};

// A node's kind is its parameter component's index in this list.
using NodeKinds = TypeList<
    Source, Output, Join, Boolean,
    primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone,
    action::mesh::Delete, action::mesh::Merge, action::mesh::Extrude, action::mesh::Duplicate, action::mesh::Split, action::mesh::Dissolve,
    action::mesh::Subdivide, action::mesh::Triangulate, action::mesh::TrisToQuads, action::mesh::Poke, action::mesh::FlipNormals, action::mesh::EdgeSplit,
    action::mesh::Inset, action::mesh::Solidify, action::mesh::Wireframe, action::mesh::Bevel, action::mesh::SmoothVertices, action::mesh::ShrinkFatten,
    action::mesh::Randomize, action::mesh::SpaceEvenly, action::mesh::RelaxEdgeLoops, action::mesh::CurveBetweenSelected, action::mesh::Circularize,
    action::mesh::MakePlanarFaces, Flatten, ToSphere, PushPull, Shear, Warp, Bend, VertexSlide, EdgeSlide, action::mesh::SnapSymmetry,
    action::mesh::Spin, action::mesh::ExtrudeRepeat, action::mesh::Bisect, action::mesh::Symmetrize, action::mesh::FillHoles,
    action::mesh::ConvexHull, action::mesh::Decimate, action::mesh::RecalculateNormals, action::mesh::Rip, action::mesh::SplitNonplanarFaces,
    action::mesh::SplitConcaveFaces, action::mesh::ConnectVertices, action::mesh::Fill, action::mesh::BridgeEdgeLoops, action::mesh::GridFill,
    action::mesh::EdgeRotate, action::mesh::BeautifyFaces, action::mesh::Unsubdivide, action::mesh::LoopCut, Knife, action::mesh::RotateUVs,
    action::mesh::ReverseUVs, action::mesh::RotateColors, action::mesh::ReverseColors, MarkSharp, ShadeSmooth, ShadeSmoothByAngle, TransformElements,
    Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, StoreSelection, SetPosition,
    MeshToPoints, InstanceOnPoints, RealizeInstances, ObjectInfo,
    FloatValue, IntegerValue, BooleanValue, VectorValue, TransformGeometry, RotationValue>;
using ValueKinds = TypeList<FloatValue, IntegerValue, BooleanValue, VectorValue, RotationValue>;
template<typename K>
concept ValueKind = InTypeList<K, ValueKinds>;

// Attribute sockets accept per-element data or broadcast constants.
// Selection sockets accept booleans and default to inline selection.
// Instances sockets reference points and their instance geometry.
enum class SocketType : uint8_t { Geometry,
                                  Selection,
                                  Attribute,
                                  Instances,
                                  Value };

enum class SocketDirection : uint8_t { In,
                                       Out };

constexpr uint8_t TypeBit(AttributeType type) { return uint8_t(1u << uint32_t(type)); }
inline constexpr uint8_t AnyType = (1u << AttributeTypeCount) - 1u;

// A parameter socket names its field by declaration index, and a nested field's member by both indices.
inline constexpr uint8_t NoField = UINT8_MAX;
inline constexpr size_t MaxParameterFields = 16u;
struct Socket {
    SocketType Type;
    SocketDirection Direction;
    uint8_t Types{}; // The attribute types a non-geometry socket carries, one TypeBit each
    bool Reads{}; // The node reads this geometry input without editing it
    const char *Name{}; // The editor label, or the type's name when null
    uint8_t Field{NoField};
    uint8_t Member{NoField};
    constexpr bool Carries(uint8_t field, uint8_t member = NoField) const { return Field == field && Member == member; }
};

namespace detail {
using enum SocketType;
using enum SocketDirection;

inline constexpr uint8_t Bool{TypeBit(AttributeType::Bool)}, Float{TypeBit(AttributeType::Float)}, Vec3{TypeBit(AttributeType::Vec3)};

// Append parameter sockets after fixed inputs.
// Sources and generators have no inputs.
inline constexpr Socket SourceSockets[]{{Geometry, Out}};
inline constexpr Socket OutputSockets[]{{Geometry, In, 0u, true}};
inline constexpr Socket JoinSockets[]{{Geometry, In}, {Geometry, In}, {Geometry, Out}};
inline constexpr Socket BooleanSockets[]{{Geometry, In, 0u, true, "Mesh 1"}, {Geometry, In, 0u, true, "Mesh 2"}, {Geometry, Out}};
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
inline constexpr Socket WholeSockets[]{{Geometry, In}, {Geometry, Out}};
// Instancing edits the points and reads the instance geometry.
inline constexpr Socket InstanceSockets[]{{Geometry, In, 0u, false, "Points"}, {Geometry, In, 0u, true, "Instance"}, {Instances, Out}};
inline constexpr Socket RealizeSockets[]{{Instances, In, 0u, true}, {Geometry, Out}};

// Return the supported socket attribute types, or zero.
template<typename F> constexpr uint8_t TypesOf() {
    if constexpr (std::same_as<F, float>) return Float;
    else if constexpr (std::same_as<F, uint32_t>) return TypeBit(AttributeType::Int);
    else if constexpr (std::same_as<F, bool>) return Bool;
    else if constexpr (std::same_as<F, vec3>) return Vec3;
    else if constexpr (std::same_as<F, quat>) return TypeBit(AttributeType::Vec4);
    else return 0u;
}

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
    else if constexpr (std::same_as<K, TransformGeometry> || std::same_as<K, Knife> || std::same_as<K, ShadeSmooth> || std::same_as<K, ShadeSmoothByAngle>) return WholeSockets;
    else if constexpr (std::same_as<K, InstanceOnPoints>) return InstanceSockets;
    else if constexpr (std::same_as<K, RealizeInstances>) return RealizeSockets;
    else if constexpr (ValueKind<K>) return ValueSockets<K>;
    else return OperatorSockets;
}

// Use attribute sockets for per-element parameters and value sockets for other scalar and vec3 fields.
template<typename K> inline constexpr field::Table<K, uint8_t> PerElement{};
template<> inline constexpr field::Table<action::mesh::Bevel, uint8_t> PerElement<action::mesh::Bevel>{{&action::mesh::Bevel::Width, Float}};
template<> inline constexpr field::Table<action::mesh::Poke, uint8_t> PerElement<action::mesh::Poke>{{&action::mesh::Poke::Offset, Float}};
template<> inline constexpr field::Table<action::mesh::Solidify, uint8_t> PerElement<action::mesh::Solidify>{{&action::mesh::Solidify::Thickness, Float}};
template<> inline constexpr field::Table<action::mesh::Wireframe, uint8_t> PerElement<action::mesh::Wireframe>{{&action::mesh::Wireframe::Thickness, Float}};
template<> inline constexpr field::Table<action::mesh::Inset, uint8_t> PerElement<action::mesh::Inset>{
    {&action::mesh::Inset::Thickness, Float},
    {&action::mesh::Inset::Depth, Float},
};
template<> inline constexpr field::Table<action::mesh::ShrinkFatten, uint8_t> PerElement<action::mesh::ShrinkFatten>{{&action::mesh::ShrinkFatten::Distance, Float}};
template<> inline constexpr field::Table<action::mesh::SmoothVertices, uint8_t> PerElement<action::mesh::SmoothVertices>{{&action::mesh::SmoothVertices::Factor, Float}};
template<> inline constexpr field::Table<action::mesh::Randomize, uint8_t> PerElement<action::mesh::Randomize>{{&action::mesh::Randomize::Amount, Float}};
template<> inline constexpr field::Table<InstanceOnPoints, uint8_t> PerElement<InstanceOnPoints>{
    {&InstanceOnPoints::Scale, Vec3},
    {&InstanceOnPoints::Rotation, TypeBit(AttributeType::Vec4)},
};
template<> inline constexpr field::Table<VectorMath, uint8_t> PerElement<VectorMath>{
    {&VectorMath::B, Vec3},
    {&VectorMath::Scale, Float},
};
template<> inline constexpr field::Table<FloatMath, uint8_t> PerElement<FloatMath>{{&FloatMath::B, Float}};
template<> inline constexpr field::Table<Compare, uint8_t> PerElement<Compare>{{&Compare::B, Float | Vec3}};
template<> inline constexpr field::Table<BooleanMath, uint8_t> PerElement<BooleanMath>{{&BooleanMath::B, Bool}};

// Order sockets as fixed inputs, parameter inputs, then fixed outputs.
// Value nodes expose their parameter as an output.
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
            constexpr auto types = TypesOf<F>();
            if constexpr (types != 0u) {
                constexpr auto per_element = PerElement<K>.ByIndex[I];
                all[n++] = {per_element ? Attribute : Value, In, per_element ? per_element : types, false, field::Label<K, I>.c_str(), uint8_t(I)};
            } else if constexpr (field::Walkable<F>) {
                field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                    constexpr auto nested_types = TypesOf<std::remove_cvref_t<decltype(nested)>>();
                    if constexpr (nested_types != 0u) {
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
template<typename... Ks> constexpr auto PerKind(TypeList<Ks...>, auto &&value) {
    return std::array<uint16_t, sizeof...(Ks)>{uint16_t(value.template operator()<Ks>())...};
}
} // namespace detail

inline constexpr auto SocketCount = detail::PerKind(NodeKinds{}, []<typename K> { return detail::SocketsOf<K>().size(); });
inline constexpr auto InputCount = detail::PerKind(NodeKinds{}, []<typename K> {
    return std::ranges::count(detail::SocketsOf<K>(), SocketDirection::In, &Socket::Direction);
});

inline constexpr uint16_t KindCount = uint16_t(SocketCount.size());
inline constexpr size_t MaxInputs = std::ranges::max(InputCount);

// Each kind's sockets start at SocketFirst[kind] in Sockets, inputs first.
inline constexpr auto SocketFirst = [] {
    std::array<uint16_t, KindCount + 1u> first{};
    for (uint16_t k = 0u; k < KindCount; ++k) first[k + 1u] = first[k] + SocketCount[k];
    return first;
}();

inline constexpr auto Sockets = []<typename... Ks>(TypeList<Ks...>) {
    std::array<Socket, SocketFirst.back()> all{};
    auto *it = all.begin();
    ((it = std::ranges::copy(detail::SocketsOf<Ks>(), it).out), ...);
    return all;
}(NodeKinds{});

constexpr std::span<const Socket> KindSockets(uint16_t kind) { return std::span{Sockets}.subspan(SocketFirst[kind], SocketCount[kind]); }

// Evaluate attribute nodes on each consumer geometry input.
constexpr bool IsAttributeKind(uint16_t kind) { return KindSockets(kind).front().Type == SocketType::Attribute; }

constexpr std::pair<uint16_t, uint16_t> ParameterRange(uint16_t kind) {
    return {uint16_t(std::ranges::count(KindSockets(kind).first(InputCount[kind]), NoField, &Socket::Field)), InputCount[kind]};
}

template<typename K> inline constexpr uint16_t KindOf = []<typename... Ks>(TypeList<Ks...>) {
    uint16_t index = 0u, found = KindCount;
    ((found = std::same_as<K, Ks> ? index : found, ++index), ...);
    return found;
}(NodeKinds{});

// Accept matching socket and attribute types, values as attributes, and booleans as selections.
// Output nodes also accept instances.
constexpr bool Links(Socket from, Socket to, uint16_t to_kind) {
    if (from.Type == SocketType::Instances) return to.Type == SocketType::Instances || to_kind == KindOf<Output>;
    const bool typed = from.Type == to.Type || ((from.Type == SocketType::Attribute || from.Type == SocketType::Value) && to.Type == SocketType::Selection) || (from.Type == SocketType::Value && to.Type == SocketType::Attribute);
    return typed && (from.Type == SocketType::Geometry || (from.Types & to.Types));
}

void VisitKind(uint16_t kind, auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) {
        uint16_t index = 0u;
        (void)((index++ == kind && (f.template operator()<Ks>(), true)) || ...);
    }(NodeKinds{});
}

void ForEachKind(auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) { (f.template operator()<Ks>(), ...); }(NodeKinds{});
}

inline constexpr uint32_t NoNode = UINT32_MAX;
struct NodeRecord {
    state::Entity Entity;
    uint16_t Kind;
};

// The output socket an input reads, or NoNode when unlinked.
struct SocketLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
};

// Persistent on the mesh entity.
struct GeometryProcedure {
    std::vector<NodeRecord> Nodes; // In authoring order
    std::vector<SocketLink> Inputs; // Each node's input sockets, in node order
    uint32_t OutputNode{NoNode}; // Output node with the result input
    uint32_t ActiveNode{NoNode}; // The node the editor's parameter panel edits
};

inline uint32_t InputFirst(const GeometryProcedure &procedure, uint32_t node) {
    uint32_t first = 0u;
    for (uint32_t n = 0u; n < node; ++n) first += InputCount[procedure.Nodes[n].Kind];
    return first;
}

inline std::vector<uint32_t> InputFirsts(const GeometryProcedure &procedure) {
    std::vector<uint32_t> first{0u};
    for (const auto &node : procedure.Nodes) first.push_back(first.back() + InputCount[node.Kind]);
    return first;
}

// Mark upstream nodes accepted by follows, using each node input offset.
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

// Return the first geometry socket in the requested direction for link insertion, or none.
inline std::optional<uint16_t> MainSocket(uint16_t kind, SocketDirection direction) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Type == SocketType::Geometry && sockets[s].Direction == direction) return s;
    }
    return {};
}

// Permit insertion when the main sockets match both link endpoints and the resulting links are acyclic.
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

// Persist published instance entities grouped by parent object.
struct ProcedureInstances {
    std::vector<state::Entity> Entities;
};

// Persist the mesh entity owning geometry generated for procedure instances.
// The entity has no object, transform, or name.
// Destroy it when the procedure stops instancing its geometry or is removed.
struct ProcedureOwnedMesh {
    state::Entity MeshEntity;
};

struct NodeLayout {
    uvec2 Position; // The top left grid cell
};

struct PublishSelection {}; // Persistent on a node a viewport operator added, which selects its output like the operator does.
struct NodeSelected {}; // Persistent on a node the editor has selected.

// Faulted nodes pass through their input, as do unchanged nodes and nodes without geometry.
enum class NodeFault : uint8_t {
    Selection, // The selection does not fit the operation
    Parameter, // A parameter is invalid
    Attribute, // An attribute input is missing or has the wrong type or domain
    NotClosed, // A boolean input is not a closed surface
    Empty, // A boolean produced no geometry
    NoMesh, // An Object Info node names no mesh it can read
};

struct NodeError {
    NodeFault Fault;
};

struct CookPlan {
    std::vector<uint32_t> Order; // Every node after the nodes it reads
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input and first socket slot, then the totals
    std::vector<uint32_t> Consumers; // The inputs reading each socket slot
};
} // namespace procedure

template<> inline constexpr field::SpecTable<procedure::Boolean> Specs<procedure::Boolean>{
    {&procedure::Boolean::Operation, {.Help = "Set operation on the two closed input meshes. Difference subtracts the second mesh from the first."}},
    {&procedure::Boolean::Frame1, {.Help = "Transform from Mesh 1 positions into the space the operation applies in"}},
    {&procedure::Boolean::Frame2, {.Help = "Transform from Mesh 2 positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::Normal> Specs<procedure::Normal>{{&procedure::Normal::Domain, {.Help = "Element domain of the emitted normals: vertices or faces"}}};
template<> inline constexpr field::SpecTable<procedure::Index> Specs<procedure::Index>{{&procedure::Index::Domain, {.Help = "Element domain of the emitted indices"}}};
template<> inline constexpr field::SpecTable<procedure::NamedAttribute> Specs<procedure::NamedAttribute>{
    {&procedure::NamedAttribute::Type, {.Help = "Value type of the attribute to read"}},
    {&procedure::NamedAttribute::Name, {.Help = "Name of the attribute to read. A missing attribute faults the node."}},
};
template<> inline constexpr field::SpecTable<procedure::VectorMath> Specs<procedure::VectorMath>{
    {&procedure::VectorMath::Operation, {.Help = "Operation on A and B. Dot, Length, and Distance give a float, and Scale multiplies A by Scale."}},
    {&procedure::VectorMath::B, {.Help = "Second operand while the B input is unlinked"}},
    {&procedure::VectorMath::Scale, {.Help = "Factor for the Scale operation while the Scale input is unlinked"}},
};
template<> inline constexpr field::SpecTable<procedure::FloatMath> Specs<procedure::FloatMath>{
    {&procedure::FloatMath::Operation, {.Help = "Operation on A and B. Unary operations such as Abs, Sqrt, Sin, and Cos ignore B, and Sin and Cos take radians."}},
    {&procedure::FloatMath::B, {.Help = "Second operand while the B input is unlinked"}},
};
template<> inline constexpr field::SpecTable<procedure::Compare> Specs<procedure::Compare>{
    {&procedure::Compare::Operation, {.Help = "Comparison of A against B"}},
    {&procedure::Compare::B, {.Help = "Value compared against A while the B input is unlinked"}},
    {&procedure::Compare::Epsilon, {.Help = "Largest difference that Equal counts as equal"}},
    {&procedure::Compare::Component, {.Help = "Vector component to compare. Float operands ignore it."}},
};
template<> inline constexpr field::SpecTable<procedure::BooleanMath> Specs<procedure::BooleanMath>{
    {&procedure::BooleanMath::Operation, {.Help = "Logical operation on A and B. Not ignores B."}},
    {&procedure::BooleanMath::B, {.Help = "Second operand while the B input is unlinked"}},
};
template<> inline constexpr field::SpecTable<procedure::StoreNamedAttribute> Specs<procedure::StoreNamedAttribute>{
    {&procedure::StoreNamedAttribute::Name, {.Help = "Name of the attribute to write. An existing attribute with this name is replaced."}},
    {&procedure::StoreNamedAttribute::Domain, {.Help = "Element domain for a constant value. A linked attribute value keeps its own domain."}},
};
template<> inline constexpr field::SpecTable<procedure::StoreSelection> Specs<procedure::StoreSelection>{{&procedure::StoreSelection::Name, {.Help = "Name of the Boolean attribute to write. An existing attribute with this name is replaced."}}};
template<> inline constexpr field::SpecTable<procedure::FloatValue> Specs<procedure::FloatValue>{{&procedure::FloatValue::Value, {.Help = "Float the node outputs"}}};
template<> inline constexpr field::SpecTable<procedure::IntegerValue> Specs<procedure::IntegerValue>{{&procedure::IntegerValue::Value, {.Help = "Unsigned integer the node outputs"}}};
template<> inline constexpr field::SpecTable<procedure::BooleanValue> Specs<procedure::BooleanValue>{{&procedure::BooleanValue::Value, {.Help = "Boolean the node outputs"}}};
template<> inline constexpr field::SpecTable<procedure::VectorValue> Specs<procedure::VectorValue>{{&procedure::VectorValue::Value, {.Help = "Vector the node outputs"}}};
template<> inline constexpr field::SpecTable<procedure::RotationValue> Specs<procedure::RotationValue>{{&procedure::RotationValue::Value, {.Help = "Rotation the node outputs"}}};
template<> inline constexpr field::SpecTable<procedure::SetPosition> Specs<procedure::SetPosition>{{&procedure::SetPosition::Offset, {.Help = "Offset each selected vertex by the Position input instead of replacing its position"}}};
template<> inline constexpr field::SpecTable<procedure::TransformElements> Specs<procedure::TransformElements>{
    {&procedure::TransformElements::Delta, {.Help = "Translation, rotation, and scale applied to the selected vertices around Pivot"}},
    {&procedure::TransformElements::Pivot, {.Help = "Center of rotation and scale in Frame space, in local units"}},
    {&procedure::TransformElements::Frame, {.Help = "Transform from mesh-local positions into the space Delta applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::TransformGeometry> Specs<procedure::TransformGeometry>{{&procedure::TransformGeometry::Delta, {.Help = "Translation, rotation, and scale applied to the whole geometry about the mesh origin"}}};
template<> inline constexpr field::SpecTable<procedure::ToSphere> Specs<procedure::ToSphere>{
    {&procedure::ToSphere::Factor, {.Min = 0, .Max = 1, .Speed = 0.01f}},
    {&procedure::ToSphere::Center, {.Speed = 0.01f, .Help = "Sphere center in Frame space"}},
    {&procedure::ToSphere::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::PushPull> Specs<procedure::PushPull>{
    {&procedure::PushPull::Center, {.Speed = 0.01f, .Help = "Point the vertices move toward, in Frame space"}},
    {&procedure::PushPull::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::Shear> Specs<procedure::Shear>{
    {&procedure::Shear::Angle, {.Min = -2 * std::numbers::pi_v<float>, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians}},
    {&procedure::Shear::Center, {.Speed = 0.01f, .Help = "Point on the shear plane, in Frame space"}},
    {&procedure::Shear::Direction, {.Speed = 0.01f, .Help = "Displacement direction in Frame space"}},
    {&procedure::Shear::Normal, {.Speed = 0.01f, .Help = "Shear plane normal in Frame space"}},
    {&procedure::Shear::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::Warp> Specs<procedure::Warp>{
    {&procedure::Warp::Angle, {.Unit = FieldUnit::Radians}},
    {&procedure::Warp::OffsetAngle, {.Unit = FieldUnit::Radians}},
    {&procedure::Warp::Center, {.Speed = 0.01f, .Help = "Warp center in Frame space"}},
    {&procedure::Warp::Orientation, {.Help = "Rotation of the warp plane in Frame space"}},
    {&procedure::Warp::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::Bend> Specs<procedure::Bend>{
    {&procedure::Bend::Angle, {.Unit = FieldUnit::Radians}},
    {&procedure::Bend::OffsetAngle, {.Unit = FieldUnit::Radians}},
    {&procedure::Bend::Center, {.Speed = 0.01f, .Help = "Bend start in Frame space"}},
    {&procedure::Bend::Orientation, {.Help = "Rotation of the bend plane in Frame space"}},
    {&procedure::Bend::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::VertexSlide> Specs<procedure::VertexSlide>{
    {&procedure::VertexSlide::Direction, {.Speed = 0.01f, .Help = "Direction in Frame space that chooses each vertex's edge"}},
    {&procedure::VertexSlide::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::EdgeSlide> Specs<procedure::EdgeSlide>{
    {&procedure::EdgeSlide::Direction, {.Speed = 0.01f, .Help = "Direction in Frame space that chooses the positive side"}},
    {&procedure::EdgeSlide::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::Flatten> Specs<procedure::Flatten>{
    {&procedure::Flatten::Factor, {.Min = 0, .Max = 1}},
    {&procedure::Flatten::ViewNormal, {.Speed = 0.01f, .Help = "Flattening direction of the View method, in Frame space"}},
    {&procedure::Flatten::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::MarkSharp> Specs<procedure::MarkSharp>{{&procedure::MarkSharp::Sharp, {.Help = "Mark the elements sharp, or clear their marks"}}};
template<> inline constexpr field::SpecTable<procedure::Knife> Specs<procedure::Knife>{
    {&procedure::Knife::Start, {.Speed = 1.f, .Help = "First end of the cut segment, in view pixels from the top left"}},
    {&procedure::Knife::End, {.Speed = 1.f, .Help = "Second end of the cut segment, in view pixels from the top left"}},
    {&procedure::Knife::Extent, {.Min = 1, .Speed = 1.f, .Help = "Width and height of the view in pixels"}},
    {&procedure::Knife::ViewProjection, {.Help = "Transform from Frame space into the view's clip space"}},
    {&procedure::Knife::Frame, {.Help = "Transform from mesh-local positions into the space ViewProjection projects"}},
};
template<> inline constexpr field::SpecTable<procedure::ShadeSmooth> Specs<procedure::ShadeSmooth>{{&procedure::ShadeSmooth::Smooth, {.Help = "Shade smooth and clear sharp marks, or shade flat"}}};
template<> inline constexpr field::SpecTable<procedure::ShadeSmoothByAngle> Specs<procedure::ShadeSmoothByAngle>{{&procedure::ShadeSmoothByAngle::Angle, {.Min = 0, .Max = std::numbers::pi, .Unit = FieldUnit::Radians, .Help = "Largest angle between adjacent face normals that shades smooth"}}};
template<> inline constexpr field::SpecTable<procedure::InstanceOnPoints> Specs<procedure::InstanceOnPoints>{
    {&procedure::InstanceOnPoints::Scale, {.Help = "Scale of each instance while the Scale input is unlinked"}},
    {&procedure::InstanceOnPoints::Rotation, {.Help = "Rotation of each instance while the Rotation input is unlinked"}},
};
template<> inline constexpr field::SpecTable<procedure::ObjectInfo> Specs<procedure::ObjectInfo>{{&procedure::ObjectInfo::Object, {.Help = "Object whose mesh the node emits. The object this procedure edits is not allowed."}}};
