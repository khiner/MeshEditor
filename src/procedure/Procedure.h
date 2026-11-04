#pragma once

#include "gpu/InteractionMode.h"
#include "procedure/InlineSelection.h"
#include "procedure/Network.h"

#include "TypeList.h"
#include "action/Mesh.h"
#include "gpu/AttributeNode.h"
#include "gpu/Transform.h"
#include "mesh/AttributeFields.h"
#include "mesh/MeshTopology.h"
#include "mesh/Minkowski.h"
#include "mesh/NamedAttribute.h"
#include "mesh/PrimitiveType.h"
#include "procedure/ProcedureSchema.h"
#include "project/store/Hash.h"

namespace procedure {
struct Source {};
struct Join {};

// Combine two closed meshes or two regions on one plane into Mesh 1's space, which Space's world transform places.
// The world transform of the object Mesh 2 came from places Mesh 2, and Mesh 2 from this procedure's mesh stays in place relative to Mesh 1.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
    state::Entity Space{state::Null};
};

// Both inputs are closed surfaces in this procedure's mesh space, or regions on parallel planes with Mesh 2's through the origin.
struct Minkowski {
    MinkowskiOperation Operation{MinkowskiOperation::Sum};
};

// Emit the surface where the Value field, positive inside, crosses Level, marched over the box from Min to Max.
// Value evaluates on a body-centered lattice spaced at most EdgeLength apart, and the surface closes at the box.
struct LevelSet {
    vec3 Min{-1, -1, -1}, Max{1, 1, 1};
    float EdgeLength{.1f};
    float Level{};
};

using Generators = TypeList<primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::QuadSphere, primitive::Torus, primitive::Cylinder>;
template<typename K>
concept Generator = InTypeList<K, Generators>;

struct Position {};

struct Normal {
    AttributeDomain Domain{AttributeDomain::Vertex};
};

struct Curvature {
    CurvatureKind Kind{CurvatureKind::Mean};
};

// Select the edges with fewer than two faces, the vertices on them, or the faces touching them.
struct Boundary {
    AttributeDomain Domain{AttributeDomain::Edge};
};
// Select the vertices on a side of the plane through Point with normal Normal, or on it within Tolerance, or the edges or faces whose vertices all are.
struct PlaneSide {
    vec3 Point{}, Normal{0, 0, 1};
    PlaneSideKind Side{PlaneSideKind::On};
    float Tolerance{PlaneSideTolerance};
    AttributeDomain Domain{AttributeDomain::Vertex};
};
// Select the connected components holding the selected elements.
struct Linked {};
// Grow the selection by Steps rings of adjacent elements, or shrink it by as many.
enum class AdjacentMode : uint8_t { More,
                                    Less };
struct Adjacent {
    AdjacentMode Mode{AdjacentMode::More};
    uint32_t Steps{1};
};
struct EdgeLoop {};
struct EdgeRing {};

struct SelectionAttribute {};

// Count live elements only.
struct Index {
    AttributeDomain Domain{AttributeDomain::Vertex};
};

struct NamedAttribute {
    AttributeType Type{AttributeType::Float};
    AttributeName Name{};
};

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

struct SetPosition {
    bool Offset{true};
};

// Move the selected vertices, all by default, by Delta around Pivot, in the space Frame maps positions into.
// A nonzero Mirror first reflects them across the plane through Pivot with that normal.
struct TransformElements {
    Transform Delta{};
    vec3 Pivot{};
    Transform Frame{};
    vec3 Mirror{};
};

// These position operators apply in the space Frame maps mesh-local positions into.

// Blend the selected vertices toward Radius from Center, or toward their mean distance from Center for a zero radius.
struct ToSphere {
    float Factor{1.f};
    vec3 Center{};
    float Radius{0.f};
    Transform Frame{};
};
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
// Extrude the selection Steps times, each step rotating Angle over Steps about Axis through Center and moving the displacement over Steps.
// The displacement is Offset Scale along the region's normal or each element's normal, or Offset times Offset Scale.
struct Extrude {
    action::mesh::ExtrudeMode Mode{};
    vec3 Offset{};
    float OffsetScale{};
    bool AlongNormals{true};
    uint32_t Steps{1};
    float Angle{};
    vec3 Axis{0, 0, 1}, Center{};
};
struct Bisect {
    vec3 Point{}, Normal{1, 0, 0};
    bool ClearInner{}, ClearOuter{};
};
struct SubdivisionSurface {
    uint32_t Levels{1};
};
// Refine the closed mesh with Cuts new vertices per edge onto smooth patches through its shading normals, creased where faces shade flat or edges are sharp.
struct SmoothSurface {
    uint32_t Cuts{2};
};
// Emit the outline of the mesh seen along Normal as a region on the plane through Point, one face per loop, in place of the mesh.
struct Project {
    vec3 Point{}, Normal{0, 0, 1};
};
// AutoRange computes Min and Max from the selected vertices after OffsetAngle.
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
// Even uses the edge length of the selected Reference vertex, or else the first selected vertex.
struct VertexSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    uint32_t Reference{InvalidOffset};
    bool Even{false}, Flipped{false}, Clamp{true};
    Transform Frame{};
};
// Slide selected edge chains along adjacent faces toward the side Direction selects at the reference vertex.
// The reference vertex is the selected Reference, or else the first selected vertex, and Even uses its rail length.
struct EdgeSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    uint32_t Reference{InvalidOffset};
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

// How every face shades: flat, smooth with no sharp marks, or smooth with each edge marked sharp whose faces' normals differ by more than Angle.
enum class ShadeMode : uint8_t { Flat,
                                 Smooth,
                                 ByAngle };
struct Shade {
    ShadeMode Mode{ShadeMode::Smooth};
    float Angle{std::numbers::pi_v<float> / 6.f};
};

// Instances use face-less point records with per-point rotation and scale attributes.
inline constexpr std::string_view InstanceRotation = ".instance.rotation", InstanceScale = ".instance.scale";

struct MeshToPoints {};

struct InstanceOnPoints {
    state::Entity Prototype{state::Null};
    vec3 Scale{1, 1, 1};
    quat Rotation{1, 0, 0, 0};
};

struct RealizeInstances {};

// Emit the object's mesh in its local space, or nothing for a missing mesh or this procedure's own mesh.
struct ObjectInfo {
    state::Entity Object{state::Null};
};

template<typename K> inline constexpr auto ObjectInput = nullptr;
template<> inline constexpr auto ObjectInput<ObjectInfo> = &ObjectInfo::Object;
template<> inline constexpr auto ObjectInput<Boolean> = &Boolean::Space;
template<> inline constexpr auto ObjectInput<InstanceOnPoints> = &InstanceOnPoints::Prototype;

struct Subnet {};

// A node's kind is its parameter component's index in this list.
using NodeKinds = TypeList<
    Source, Join, Boolean, Minkowski,
    primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::QuadSphere, primitive::Torus, primitive::Cylinder, LevelSet,
    action::mesh::Delete, action::mesh::Merge, Extrude, action::mesh::Duplicate, action::mesh::Split, action::mesh::Dissolve,
    action::mesh::Subdivide, SubdivisionSurface, SmoothSurface, action::mesh::SplitFaces, action::mesh::TrisToQuads, action::mesh::Poke, action::mesh::FlipNormals, action::mesh::EdgeSplit,
    action::mesh::Inset, action::mesh::Solidify, action::mesh::Wireframe, action::mesh::Bevel, action::mesh::SmoothVertices, action::mesh::ShrinkFatten,
    action::mesh::Randomize, action::mesh::SpaceEvenly, action::mesh::RelaxEdgeLoops, action::mesh::CurveBetweenSelected, action::mesh::Circularize,
    action::mesh::MakePlanarFaces, Flatten, ToSphere, PushPull, Shear, Warp, Bend, VertexSlide, EdgeSlide, action::mesh::SnapSymmetry,
    Bisect, Project,
    action::mesh::ConvexHull, action::mesh::Simplify, action::mesh::RecalculateNormals, action::mesh::ConnectVertices, action::mesh::Fill, action::mesh::BridgeEdgeLoops, action::mesh::GridFill,
    action::mesh::EdgeRotate, action::mesh::BeautifyFaces, action::mesh::Unsubdivide, Knife, action::mesh::CycleCornerAttribute, MarkSharp, Shade, TransformElements,
    Position, Normal, Curvature, Boundary, PlaneSide, Linked, Adjacent, EdgeLoop, EdgeRing, SelectionAttribute, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, StoreSelection, SetPosition,
    MeshToPoints, InstanceOnPoints, RealizeInstances, ObjectInfo,
    FloatValue, IntegerValue, BooleanValue, VectorValue, RotationValue, Subnet>;
using ValueKinds = TypeList<FloatValue, IntegerValue, BooleanValue, VectorValue, RotationValue>;
template<typename K>
concept ValueKind = InTypeList<K, ValueKinds>;

// Selection sockets accept booleans and default to inline selection.
enum class SocketType : uint8_t { Geometry,
                                  Selection,
                                  Attribute,
                                  Instances,
                                  Value,
                                  Reference };

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
    constexpr bool Carries(uint8_t field, uint8_t member) const { return Field == field && Member == member; }
};

namespace detail {
using enum SocketType;
using enum SocketDirection;

inline constexpr uint8_t Bool{TypeBit(AttributeType::Bool)}, Float{TypeBit(AttributeType::Float)}, Vec3{TypeBit(AttributeType::Vec3)};

inline constexpr Socket SourceSockets[]{{Geometry, Out}};
inline constexpr Socket JoinSockets[]{{Geometry, In}, {Geometry, In}, {Geometry, Out}};
inline constexpr Socket BooleanSockets[]{{Geometry, In, 0u, true, "Mesh 1"}, {Geometry, In, 0u, true, "Mesh 2"}, {Geometry, Out}};
inline constexpr Socket OperatorSockets[]{{Geometry, In}, {Selection, In, Bool}, {Geometry, Out}};
inline constexpr Socket PositionSockets[]{{Attribute, Out, Vec3}};
inline constexpr Socket LevelSetSockets[]{{Attribute, In, Float, false, "Value"}, {Geometry, Out}};
inline constexpr Socket CurvatureSockets[]{{Attribute, Out, Float}};
inline constexpr Socket SelectionOutputSockets[]{{Attribute, Out, Bool}};
inline constexpr Socket SelectionFieldSockets[]{{Attribute, In, Bool, false, "Selection"}, {Attribute, Out, Bool}};
inline constexpr Socket SelectionInputSockets[]{{Selection, In, Bool}, {Attribute, Out, Bool}};
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
inline constexpr Socket InstanceSockets[]{{Geometry, In, 0u, false, "Points"}, {Geometry, In, 0u, true, "Instance"}, {Instances, Out}};
inline constexpr Socket RealizeSockets[]{{Instances, In, 0u, true}, {Geometry, Out}};

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
    if constexpr (std::same_as<K, Subnet>) return std::span<const Socket>{};
    else if constexpr (std::same_as<K, Source> || std::same_as<K, ObjectInfo> || Generator<K>) return SourceSockets;
    else if constexpr (std::same_as<K, Join>) return JoinSockets;
    else if constexpr (std::same_as<K, Boolean> || std::same_as<K, Minkowski>) return BooleanSockets;
    else if constexpr (std::same_as<K, Position> || std::same_as<K, Normal>) return PositionSockets;
    else if constexpr (std::same_as<K, Curvature>) return CurvatureSockets;
    else if constexpr (std::same_as<K, Boundary> || std::same_as<K, PlaneSide>) return SelectionOutputSockets;
    else if constexpr (std::same_as<K, Linked> || std::same_as<K, Adjacent> || std::same_as<K, EdgeLoop> || std::same_as<K, EdgeRing>) return SelectionFieldSockets;
    else if constexpr (std::same_as<K, SelectionAttribute>) return SelectionInputSockets;
    else if constexpr (std::same_as<K, LevelSet>) return LevelSetSockets;
    else if constexpr (std::same_as<K, Index>) return IndexSockets;
    else if constexpr (std::same_as<K, NamedAttribute>) return NamedAttributeSockets;
    else if constexpr (std::same_as<K, VectorMath>) return VectorMathSockets;
    else if constexpr (std::same_as<K, FloatMath>) return FloatMathSockets;
    else if constexpr (std::same_as<K, Compare>) return CompareSockets;
    else if constexpr (std::same_as<K, BooleanMath>) return BooleanMathSockets;
    else if constexpr (std::same_as<K, StoreNamedAttribute>) return StoreSockets;
    else if constexpr (std::same_as<K, SetPosition>) return SetPositionSockets;
    else if constexpr (std::same_as<K, MeshToPoints>) return PointsSockets;
    else if constexpr (std::same_as<K, Knife> || std::same_as<K, Shade> || std::same_as<K, Project> || std::same_as<K, SubdivisionSurface> || std::same_as<K, SmoothSurface>) return WholeSockets;
    else if constexpr (std::same_as<K, InstanceOnPoints>) return InstanceSockets;
    else if constexpr (std::same_as<K, RealizeInstances>) return RealizeSockets;
    else if constexpr (ValueKind<K>) return ValueSockets<K>;
    else return OperatorSockets;
}

// Use attribute sockets for per-element parameters and value sockets for other scalar and vec3 fields.
template<typename K> inline constexpr field::Table<K, uint8_t> PerElement{};
template<> inline constexpr field::Table<Extrude, uint8_t> PerElement<Extrude>{
    {&Extrude::Offset, Vec3},
    {&Extrude::OffsetScale, Float},
};
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

void VisitSocketParameter(auto &parameters, Socket socket, auto &&visit) {
    if constexpr (field::Walkable<std::remove_cvref_t<decltype(parameters)>>) {
        field::ForEach(parameters, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
            if (socket.Field != I) return;
            using F = std::remove_cvref_t<decltype(member)>;
            if constexpr (detail::TypesOf<F>() != 0u) {
                if (socket.Member == NoField) visit(member);
            } else if constexpr (field::Walkable<F>) {
                field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                    if constexpr (detail::TypesOf<std::remove_cvref_t<decltype(nested)>>() != 0u)
                        if (socket.Member == J) visit(nested);
                });
            }
        });
    }
}
void AssignSocketParameter(auto &parameters, Socket socket, const AttributeValue &words) {
    VisitSocketParameter(parameters, socket, [&](auto &value) { std::memcpy(&value, words.data(), sizeof(value)); });
}

// An attribute node's first output is an attribute, and the node evaluates on each consumer's geometry input.
constexpr bool IsAttributeKind(uint16_t kind) { return SocketCount[kind] > InputCount[kind] && KindSockets(kind)[InputCount[kind]].Type == SocketType::Attribute; }

constexpr std::pair<uint16_t, uint16_t> ParameterRange(uint16_t kind) {
    return {uint16_t(std::ranges::count(KindSockets(kind).first(InputCount[kind]), NoField, &Socket::Field)), InputCount[kind]};
}

template<typename K> inline constexpr uint16_t KindOf = []<typename... Ks>(TypeList<Ks...>) {
    uint16_t index = 0u, found = KindCount;
    ((found = std::same_as<K, Ks> ? index : found, ++index), ...);
    return found;
}(NodeKinds{});
// The input socket carrying the parameter field M of its node kind, or the member of M the nested pointer selects.
template<auto M, auto... Nested> inline constexpr uint16_t ParameterSocket = [] {
    constexpr uint8_t member = [] { if constexpr (sizeof...(Nested) == 0u) return NoField; else return uint8_t(field::IndexOf<Nested...[0]>); }();
    constexpr auto sockets = KindSockets(KindOf<field::Owner<M>>);
    constexpr auto socket = uint16_t(std::ranges::find_if(sockets, [](Socket s) { return s.Carries(uint8_t(field::IndexOf<M>), member); }) - sockets.begin());
    static_assert(socket < sockets.size(), "The field is not a parameter socket of its node kind");
    return socket;
}();

struct HandleSpec {
    uint16_t Kind;
    const char *Label;
    uint64_t Fields;
    bool Reads(Socket socket) const { return socket.Field < 64 && (Fields & (uint64_t{1} << socket.Field)); }
};
template<auto... Members> inline constexpr uint64_t HandleFields = ((uint64_t{1} << field::IndexOf<Members>) | ...);
// Indexed by HandleParameter.
inline constexpr std::array HandleSpecs{
    HandleSpec{KindOf<TransformElements>, "Transform", HandleFields<&TransformElements::Delta>},
    HandleSpec{KindOf<Extrude>, "Offset", HandleFields<&Extrude::Offset, &Extrude::OffsetScale, &Extrude::AlongNormals>},
    HandleSpec{KindOf<Extrude>, "Offset Scale", HandleFields<&Extrude::OffsetScale>},
    HandleSpec{KindOf<action::mesh::Bevel>, "Width", HandleFields<&action::mesh::Bevel::Width>},
    HandleSpec{KindOf<action::mesh::Inset>, "Thickness", HandleFields<&action::mesh::Inset::Thickness>},
    HandleSpec{KindOf<action::mesh::Inset>, "Depth", HandleFields<&action::mesh::Inset::Depth>},
};

// Accept matching socket and attribute types, values as attributes, and booleans as selections.
constexpr bool Links(Socket from, Socket to) {
    if (from.Type == SocketType::Instances) return to.Type == SocketType::Instances;
    const bool typed = from.Type == to.Type || ((from.Type == SocketType::Attribute || from.Type == SocketType::Value) && to.Type == SocketType::Selection) || (from.Type == SocketType::Value && to.Type == SocketType::Attribute);
    return typed && (from.Type == SocketType::Geometry || (from.Types & to.Types));
}

void VisitKind(uint16_t kind, auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) {
        uint16_t index = 0u;
        (void)((index++ == kind && (f.template operator()<Ks>(), true)) || ...);
    }(NodeKinds{});
}

bool PrepareOperationHandle(state::Registry &, state::Entity);
uint32_t SelectionInput(state::Registry &, state::Entity node);
void SetMeshSelection(state::Registry &, state::Entity mesh, const InlineSelection *);

void ForEachKind(auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) { (f.template operator()<Ks>(), ...); }(NodeKinds{});
}

inline constexpr uint32_t NoNode = UINT32_MAX;

// Next is the node after this one in authoring order, or null for the last.
struct ProcedureNode {
    state::Entity Procedure{state::Null};
    state::Entity Next{state::Null};
    uint16_t Kind{};
};

struct PortValue {
    AttributeValue Words{};
    AttributeType Type{AttributeType::Float};
    bool operator==(const PortValue &) const = default;
};
struct SubnetPort {
    uint16_t Id{};
    std::string Name;
    SocketType Type{SocketType::Geometry};
    SocketDirection Direction{SocketDirection::In};
    uint8_t Types{};
    PortBinding Binding;
    PortValue Default;
    SocketLink &Source() { return std::get<SocketLink>(Binding); }
    const SocketLink &Source() const { return std::get<SocketLink>(Binding); }
};
struct SubnetInterface {
    uint32_t NextId{};
    std::vector<SubnetPort> Ports;
};
struct NodeContainer {
    state::Entity Parent{state::Null};
};
struct SocketEdit {
    state::Entity Node;
    uint16_t Socket;
    SocketLink Source;
};
struct ResolvedSocket {
    SocketLink Source;
    std::optional<PortValue> Constant;
    bool Cycle{};
};
state::Entity ContainerOf(const state::Registry &, state::Entity node);
const SubnetPort *PortOf(const state::Registry &, state::Entity subnet, uint16_t port);
Socket SocketOf(const state::Registry &, state::Entity node, uint16_t socket);
SocketLink InputOf(const state::Registry &, state::Entity node, uint16_t socket);
void SetInput(state::Registry &, state::Entity node, uint16_t socket, SocketLink);
ResolvedSocket ResolveSocket(const state::Registry &, SocketLink, bool bypass = false, const SocketEdit * = nullptr);
std::optional<uint16_t> SubnetBypassInput(const state::Registry &, state::Entity, uint16_t output);
std::optional<PortValue> ParameterDefault(const state::Registry &, state::Entity node, uint16_t socket);
void SetParameterDefault(state::Registry &, state::Entity node, uint16_t socket, PortValue);

struct NodeInputs {
    std::vector<SocketLink> Links;
};

// The output socket carrying a node's geometry or instances, which the mesh draws when the node is displayed, or none.
constexpr std::optional<uint16_t> ResultSocket(uint16_t kind) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = InputCount[kind]; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Type == SocketType::Geometry || sockets[s].Type == SocketType::Instances) return s;
    }
    return {};
}

struct GeometryProcedure {
    state::Entity First{state::Null}, Last{state::Null};
    state::Entity DisplayNode{state::Null};
    std::optional<uint16_t> DisplayPort;
    bool operator==(const GeometryProcedure &) const = default;
};

std::optional<uint16_t> ResultSocket(const state::Registry &, state::Entity);
SocketLink DisplayLink(const state::Registry &, const GeometryProcedure &);

struct NodeBypassed {};

struct NodeRecord {
    state::Entity Entity;
    uint16_t Kind;
};
// The output socket an input reads by node index in its plan, or NoNode when unlinked.
struct NodeLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
    bool operator==(const NodeLink &) const = default;
};
struct PlanLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
    std::optional<PortValue> Constant;
    bool operator==(const PlanLink &) const = default;
};
struct CookPlan {
    uint64_t Revision{};
    std::vector<Socket> ViewSockets;
    std::vector<uint16_t> ViewSocketIds;
    std::vector<NodeRecord> Nodes; // In authoring order
    std::vector<uint32_t> ByEntity; // Node indices in entity order
    std::vector<PlanLink> Inputs; // Resolved execution inputs
    std::vector<NodeLink> Wires; // Authored inputs for editor interactions
    uint32_t DisplayNode{NoNode}, ResultNode{NoNode};
    std::vector<uint32_t> Order; // Every node after the nodes it reads
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input and first socket slot, then the totals
    std::vector<uint32_t> Consumers; // The inputs reading each socket slot
    std::vector<bool> Demanded; // Resolved ancestors of the displayed result
};
std::vector<state::Entity> NodesOf(const state::Registry &, const GeometryProcedure &);
CookPlan Plan(const state::Registry &, const GeometryProcedure &, bool bypass = false, const SocketEdit * = nullptr);
CookPlan Compile(const state::Registry &, const GeometryProcedure &);
bool ObjectReferencesAcyclic(const state::Registry &, std::span<const state::Entity> nodes, state::Entity object, state::Entity additional_reader = state::Null, std::optional<std::pair<state::Entity, state::Entity>> binding = {});
void BindDisplay(const state::Registry &, const GeometryProcedure &, CookPlan &);
CookPlan ProjectNetwork(const state::Registry &, state::Entity owner, state::Entity container);
inline uint16_t InputsOf(const CookPlan &plan, uint32_t node) { return uint16_t(plan.InputFirst[node + 1] - plan.InputFirst[node]); }
inline std::span<const Socket> NodeSockets(const CookPlan &plan, uint32_t node) {
    return plan.ViewSocketIds.empty() ? KindSockets(plan.Nodes[node].Kind) : std::span{plan.ViewSockets}.subspan(plan.SocketFirst[node], plan.SocketFirst[node + 1] - plan.SocketFirst[node]);
}
inline uint16_t SocketId(const CookPlan &plan, uint32_t node, uint16_t socket) {
    return plan.ViewSocketIds.empty() ? socket : plan.ViewSocketIds[plan.SocketFirst[node] + socket];
}
inline std::pair<uint16_t, uint16_t> ParameterRange(const CookPlan &plan, uint32_t node) {
    return plan.Nodes[node].Kind == KindOf<Subnet> ? std::pair<uint16_t, uint16_t>{InputsOf(plan, node), InputsOf(plan, node)} : ParameterRange(plan.Nodes[node].Kind);
}
std::optional<uint16_t> BypassInput(uint16_t kind);
state::Entity ActiveNodeOf(const state::Registry &, state::Entity mesh_entity);
void IndexNodes(CookPlan &);
uint32_t IndexOf(const CookPlan &, state::Entity);
uint32_t ActiveIndex(const state::Registry &, const CookPlan &);
inline state::Entity EntityOf(const CookPlan &plan, uint32_t node) { return node == NoNode ? state::Null : plan.Nodes[node].Entity; }

// Orders nodes after their sources, calling edges(visit) twice to visit each (source, reader) edge.
// Nodes in a cycle and their dependents are omitted, and ready nodes keep input order.
std::vector<uint32_t> DependencyOrder(uint32_t count, auto &&edges) {
    std::vector<uint32_t> pending(count), first(count + 1u);
    edges([&](uint32_t source, uint32_t reader) { ++pending[reader]; ++first[source + 1u]; });
    for (uint32_t n = 0; n < count; ++n) first[n + 1u] += first[n];
    std::vector<uint32_t> readers(first.back());
    auto fill = first;
    edges([&](uint32_t source, uint32_t reader) { readers[fill[source]++] = reader; });
    std::vector<uint32_t> order;
    order.reserve(count);
    for (uint32_t n = 0; n < count; ++n)
        if (!pending[n]) order.push_back(n);
    for (size_t i = 0; i < order.size(); ++i)
        for (uint32_t at = first[order[i]]; at < first[order[i] + 1u]; ++at)
            if (!--pending[readers[at]]) order.push_back(readers[at]);
    return order;
}

std::vector<bool> Upstream(std::span<const uint32_t> first, auto inputs, uint32_t node, auto &&follows) {
    std::vector<bool> marked(first.empty() ? 0u : first.size() - 1u);
    if (node == NoNode) return marked;
    marked[node] = true;
    for (std::vector<uint32_t> stack{node}; !stack.empty();) {
        const auto n = stack.back();
        stack.pop_back();
        for (uint32_t i = first[n]; i < first[n + 1u]; ++i) {
            if (const auto from = inputs[i].FromNode; from != NoNode && !marked[from] && follows(from)) {
                marked[from] = true;
                stack.push_back(from);
            }
        }
    }
    return marked;
}
inline std::vector<bool> Upstream(const CookPlan &plan, uint32_t node, auto &&follows, bool authored = false) {
    return authored ? Upstream(plan.InputFirst, std::span{plan.Wires}, node, follows) : Upstream(plan.InputFirst, std::span{plan.Inputs}, node, follows);
}

// The first geometry socket in the direction, or none.
inline std::optional<uint16_t> MainSocket(uint16_t kind, SocketDirection direction) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Type == SocketType::Geometry && sockets[s].Direction == direction) return s;
    }
    return {};
}

// Permit insertion when the main sockets match both link endpoints and the resulting links are acyclic.
inline bool Inserts(const CookPlan &plan, uint32_t node, uint32_t to_node, uint16_t to_socket) {
    const auto link = plan.Wires[plan.InputFirst[to_node] + to_socket];
    const auto kind = plan.Nodes[node].Kind;
    const auto input = MainSocket(kind, SocketDirection::In), output = MainSocket(kind, SocketDirection::Out);
    if (link.FromNode == NoNode || link.FromNode == node || to_node == node || !input || !output) return false;
    if (!Links(NodeSockets(plan, node)[*output], NodeSockets(plan, to_node)[to_socket]) || !Links(NodeSockets(plan, link.FromNode)[link.FromSocket], NodeSockets(plan, node)[*input])) return false;
    const auto all = [](uint32_t) { return true; };
    return !Upstream(plan, node, all, true)[to_node] && !Upstream(plan, link.FromNode, all, true)[node];
}

struct ProcedureInstances {
    std::vector<state::Entity> Entities;
};

// The mesh entity has no object, transform or name, and is destroyed when the procedure stops instancing its geometry or is removed.
struct ProcedureOwnedMesh {
    state::Entity MeshEntity;
};

struct NodeLayout {
    uvec2 Position; // The top left grid cell
};
inline constexpr uint32_t AppendGap{12u}; // The cells an appended node sits right of the displayed node.
// The kind's name with spaces between its words.
template<typename K> constexpr auto Title{field::detail::Spaced<field::fixed_string<state::LeafName<K>().size() + 1>{state::LeafName<K>()}>()};

struct PublishSelection {}; // Persistent on a node a viewport operator or macro added, which selects its output as the operator does.

state::Entity CreateNode(state::Registry &, uint16_t kind, uvec2 position);
// Sets the mesh entity's procedure to the root node of the kind alone, displayed, active and selected.
void RootProcedure(state::Registry &, state::Entity mesh_entity, state::Entity root, uint16_t kind);

// Faulted nodes pass through their input, as do unchanged nodes and nodes without geometry.
enum class NodeFault : uint8_t {
    Selection,
    Parameter,
    Attribute, // An attribute input is missing or has the wrong type or domain
    NotClosed,
    NotCoplanar, // A region input's face edges leave the first region's plane
    Empty,
    NoMesh, // An Object Info node refers to no mesh it can read, or an Instance input is unlinked
};

struct NodeError {
    NodeFault Fault;
};

// Persistent on a Source node: the mesh storage state, identified by its hash, in which record StoreId holds the node's geometry.
// A pending state is zero, and its record holds the geometry until the node's history step ends.
// A state without a record is empty geometry.
struct SourceState {
    store::Hash128 State{};
    uint32_t StoreId{InvalidOffset};
    bool operator==(const SourceState &) const = default;
};
// Persistent on each node of the chained display path its mesh's record was last cooked through.
// Input is the path node whose result this node ran on, or null for the root.
// The path's last node holds a pending zero state, and the record holds that node's result as it stands.
struct NodeResult {
    state::Entity Input{state::Null};
    store::Hash128 State{};
    uint32_t StoreId{InvalidOffset};
    bool operator==(const NodeResult &) const = default;
};
struct NodeExtrusion {
    store::Hash128 State{}, Selection{};
    action::mesh::ExtrudeMode Mode{};
    vec3 RegionNormal{};
    std::vector<ExtrusionSource> Vertices;
    bool operator==(const NodeExtrusion &) const = default;
};
struct NodeHandle {
    vec3 Center{}, Normal{};
    bool operator==(const NodeHandle &) const = default;
};
struct SelectionEditing {
    state::Entity Node, Mesh, Placement;
    InteractionMode PriorMode;
    Element PriorElement;
    std::optional<InlineSelection> PriorSelection;
};
struct OperationHandle {
    state::Entity Node{state::Null}, Object{state::Null};
    HandleParameter Parameter{};
};
// Persistent on a Boolean node the last cook evaluated, with the transform from Mesh 2's space into Mesh 1's that its result was cooked with.
// Operand is the object whose transform placed Mesh 2, or null for Mesh 2 in this procedure's mesh space.
// A procedure whose placing objects only moved cooks when this transform changed.
struct BooleanPlacement {
    state::Entity Operand{state::Null};
    mat4 ToFirst{1.f};
    bool operator==(const BooleanPlacement &) const = default;
};
struct StagedNode {}; // Derived on a node a preview step appended, which the stage's later steps edit.
} // namespace procedure

template<> inline constexpr field::SpecTable<procedure::Boolean> Specs<procedure::Boolean>{
    {&procedure::Boolean::Operation, {.Help = "Set operation on the two inputs, closed meshes or regions on one plane. Difference subtracts the second from the first."}},
    {&procedure::Boolean::Space, {.Help = "Object whose world transform places Mesh 1 and whose local space the result is in, or none to leave Mesh 1 in place. The object Mesh 2's geometry came from places Mesh 2."}},
};
template<> inline constexpr field::SpecTable<procedure::Minkowski> Specs<procedure::Minkowski>{{&procedure::Minkowski::Operation, {.Help = "Sum sweeps Mesh 1 by Mesh 2, and Difference erodes Mesh 1 by it. Two regions sweep in Mesh 1's plane, with Mesh 2's plane through the origin."}}};
template<> inline constexpr field::SpecTable<procedure::Normal> Specs<procedure::Normal>{{&procedure::Normal::Domain, {.Help = "Element domain of the emitted normals: vertices or faces"}}};
template<> inline constexpr field::SpecTable<procedure::LevelSet> Specs<procedure::LevelSet>{
    {&procedure::LevelSet::Min, {.Speed = 0.01f, .Help = "Lower corner of the box the surface is marched over, in local space"}},
    {&procedure::LevelSet::Max, {.Speed = 0.01f, .Help = "Upper corner of the box, in local space"}},
    {&procedure::LevelSet::EdgeLength, {.Min = 1e-4f, .Speed = 0.001f, .Help = "Largest spacing of the lattice the Value field is read on, which bounds the triangles' edge lengths"}},
    {&procedure::LevelSet::Level, {.Speed = 0.01f, .Help = "Value the surface is extracted at, with larger values inside"}},
};
template<> inline constexpr field::SpecTable<procedure::Boundary> Specs<procedure::Boundary>{{&procedure::Boundary::Domain, {.Help = "Domain of the selection: boundary edges, the vertices on them, or the faces touching them"}}};
template<> inline constexpr field::SpecTable<procedure::PlaneSide> Specs<procedure::PlaneSide>{
    {&procedure::PlaneSide::Point, {.Speed = 0.01f, .Help = "Point on the plane, in local space"}},
    {&procedure::PlaneSide::Normal, {.Speed = 0.01f, .Help = "Normal of the plane toward its positive side, in local space. A zero normal faults the node."}},
    {&procedure::PlaneSide::Side, {.Help = "Select the vertices in front of the plane, behind it, or on it"}},
    {&procedure::PlaneSide::Tolerance, {.Min = 0.f, .Speed = 1e-5f, .Help = "Distance from the plane within which a vertex lies on it, relative to the larger of one and the vertex's distance from the origin"}},
    {&procedure::PlaneSide::Domain, {.Help = "Domain of the selection: the vertices, or the edges or faces whose vertices are all selected"}},
};
template<> inline constexpr field::SpecTable<procedure::Adjacent> Specs<procedure::Adjacent>{
    {&procedure::Adjacent::Mode, {.Help = "Grow the selection by adjacent elements, or shrink it by the elements next to unselected ones"}},
    {&procedure::Adjacent::Steps, {.Min = 1, .Max = 64, .Help = "Rings of adjacent elements to add or remove"}},
};
template<> inline constexpr field::SpecTable<procedure::Curvature> Specs<procedure::Curvature>{{&procedure::Curvature::Kind, {.Help = "Gaussian curvature from each vertex's angle defect, or mean curvature from the dihedral angles of its edges"}}};
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
    {&procedure::TransformElements::Delta, {.Help = "Translation, rotation, and scale applied to the selected vertices, all by default, around Pivot"}},
    {&procedure::TransformElements::Pivot, {.Help = "Center of rotation and scale in Frame space, in local units"}},
    {&procedure::TransformElements::Frame, {.Help = "Transform from mesh-local positions into the space Delta applies in"}},
    {&procedure::TransformElements::Mirror, {.Speed = 0.01f, .Help = "Normal of the plane through Pivot the selected vertices reflect across before Delta, in Frame space. A zero normal reflects nothing."}},
};
template<> inline constexpr field::SpecTable<procedure::ToSphere> Specs<procedure::ToSphere>{
    {&procedure::ToSphere::Factor, {.Min = 0, .Max = 1, .Speed = 0.01f}},
    {&procedure::ToSphere::Center, {.Speed = 0.01f, .Help = "Sphere center in Frame space"}},
    {&procedure::ToSphere::Radius, {.Min = 0, .Speed = 0.01f, .Help = "Distance from Center the vertices blend toward, in local units. Zero takes the selected vertices' mean distance."}},
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
    {&procedure::VertexSlide::Reference, {.Help = "Vertex the even slide measures from, or the first selected vertex"}},
    {&procedure::VertexSlide::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<procedure::EdgeSlide> Specs<procedure::EdgeSlide>{
    {&procedure::EdgeSlide::Direction, {.Speed = 0.01f, .Help = "Direction in Frame space that chooses the positive side"}},
    {&procedure::EdgeSlide::Reference, {.Help = "Handle of the vertex where Direction chooses the positive side, or the maximum for the first selected vertex"}},
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
template<> inline constexpr field::SpecTable<procedure::Shade> Specs<procedure::Shade>{
    {&procedure::Shade::Mode, {.Help = "Shade every face flat, smooth with no sharp marks, or smooth with sharp marks where faces meet above Angle"}},
    {&procedure::Shade::Angle, {.Min = 0, .Max = std::numbers::pi, .Unit = FieldUnit::Radians, .Help = "Largest angle between adjacent face normals that shades smooth"}},
};
template<> inline constexpr field::SpecTable<procedure::InstanceOnPoints> Specs<procedure::InstanceOnPoints>{
    {&procedure::InstanceOnPoints::Prototype, {.Help = "Object hierarchy placed at each point. When empty, use the Instance geometry input."}},
    {&procedure::InstanceOnPoints::Scale, {.Help = "Scale of each instance while the Scale input is unlinked"}},
    {&procedure::InstanceOnPoints::Rotation, {.Help = "Rotation of each instance while the Rotation input is unlinked"}},
};
template<> inline constexpr field::SpecTable<procedure::ObjectInfo> Specs<procedure::ObjectInfo>{{&procedure::ObjectInfo::Object, {.Help = "Object whose mesh the node emits. The object this procedure edits is not allowed."}}};

template<> inline constexpr field::SpecTable<procedure::Extrude> Specs<procedure::Extrude>{
    {&procedure::Extrude::Mode, field::SpecOf<&action::mesh::Extrude::Mode>},
    {&procedure::Extrude::Offset, field::SpecOf<&action::mesh::Extrude::Offset>},
    {&procedure::Extrude::OffsetScale, field::SpecOf<&action::mesh::Extrude::OffsetScale>},
    {&procedure::Extrude::AlongNormals, field::SpecOf<&action::mesh::Extrude::AlongNormals>},
    {&procedure::Extrude::Steps, field::SpecOf<&action::mesh::Revolve::Steps>},
    {&procedure::Extrude::Angle, field::SpecOf<&action::mesh::Revolve::Angle>},
    {&procedure::Extrude::Axis, field::SpecOf<&action::mesh::Revolve::Axis>},
    {&procedure::Extrude::Center, field::SpecOf<&action::mesh::Revolve::Center>},
};
template<> inline constexpr field::SpecTable<procedure::SubdivisionSurface> Specs<procedure::SubdivisionSurface>{{&procedure::SubdivisionSurface::Levels, {.Min = 1, .Max = 6, .Help = "Number of subdivisions, each quadrupling the face count"}}};
template<> inline constexpr field::SpecTable<procedure::SmoothSurface> Specs<procedure::SmoothSurface>{{&procedure::SmoothSurface::Cuts, {.Min = 1, .Max = 16, .Help = "Number of new vertices along each edge"}}};
template<> inline constexpr field::SpecTable<procedure::Project> Specs<procedure::Project>{
    {&procedure::Project::Point, {.Speed = 0.01f, .Help = "Point on the plane the outline lies on, in local space"}},
    {&procedure::Project::Normal, {.Speed = 0.01f, .Help = "Direction the mesh is seen along, and the plane's normal, in local space"}},
};
template<> inline constexpr field::SpecTable<procedure::Bisect> Specs<procedure::Bisect>{
    {&procedure::Bisect::Point, field::SpecOf<&action::mesh::Bisect::Point>},
    {&procedure::Bisect::Normal, field::SpecOf<&action::mesh::Bisect::Normal>},
    {&procedure::Bisect::ClearInner, field::SpecOf<&action::mesh::Bisect::ClearInner>},
    {&procedure::Bisect::ClearOuter, field::SpecOf<&action::mesh::Bisect::ClearOuter>},
};
