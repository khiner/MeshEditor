#pragma once

#include "gpu/InteractionMode.h"
#include "graph/InlineSelection.h"
#include "graph/ItemGraph.h"

#include "TypeList.h"
#include "action/Core.h"
#include "action/Mesh.h"
#include "gpu/Transform.h"
#include "graph/BuiltinSubgraphs.h"
#include "graph/GraphSchema.h"
#include "graph/NumericValue.h"
#include "mesh/MeshTopology.h"
#include "mesh/Minkowski.h"
#include "mesh/NamedAttribute.h"
#include "mesh/PrimitiveType.h"
#include "project/store/Hash.h"

#include <map>

namespace mx {
enum class Type : uint8_t;
} // namespace mx

namespace graph {
struct Source {};
struct Join {};

// Combine Mesh 1 with each Mesh 2, in link order, into Mesh 1's space, which Space's world transform places.
// The inputs are closed meshes or regions on one plane.
// The world transform of the object a Mesh 2 came from places it, and a Mesh 2 from this graph's mesh stays in place relative to Mesh 1.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
    state::Entity Space{state::Null};
};

// Both inputs are closed surfaces in this graph's mesh space, or regions on parallel planes with Mesh 2's through the origin.
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

// Replace the named attribute with the value evaluated on Domain.
struct StoreNamedAttribute {
    AttributeName Name{};
    AttributeDomain Domain{AttributeDomain::Vertex};
};

// Replace the named Bool attribute with the selection, in a linked attribute's domain or else the edit domain of the inline selection.
struct StoreSelection {
    AttributeName Name{};
};

// Store the Value field over the geometry's layout in UV set UvSet on a Width by Height texel grid.
// A texel holds the field at the point of a triangle covering its center, or else at the nearest point of the nearest triangle within Margin texels, and zero beyond.
struct Bake {
    uint32_t UvSet{0};
    uint32_t Width{1024}, Height{1024};
    uint32_t Margin{16};
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

// Emit the object's mesh in its local space, or nothing for a missing mesh or this graph's own mesh.
struct ObjectInfo {
    state::Entity Object{state::Null};
};

// The sink of a material's node graph, whose inputs are the gltf_pbr inputs a graph computes at each surface point, in PbrInput order.
// An unlinked input reads its material's value, and a linked one replaces the value the material's factor and texture give.
struct Surface {};

template<typename K> inline constexpr auto ObjectInput = nullptr;
template<> inline constexpr auto ObjectInput<ObjectInfo> = &ObjectInfo::Object;
template<> inline constexpr auto ObjectInput<Boolean> = &Boolean::Space;
template<> inline constexpr auto ObjectInput<InstanceOnPoints> = &InstanceOnPoints::Prototype;

// A set of data types, one TypeBit each.
using DataTypes = uint16_t;
constexpr DataTypes TypeBit(DataType type) { return DataTypes(1u << uint32_t(type)); }
inline constexpr DataTypes NumericTypes = TypeBit(DataType::Float) | TypeBit(DataType::Vec2) | TypeBit(DataType::Vec3) | TypeBit(DataType::Vec4) | TypeBit(DataType::Color3) | TypeBit(DataType::Color4) | TypeBit(DataType::Int) | TypeBit(DataType::Bool);
inline constexpr DataTypes MatrixTypes = TypeBit(DataType::Matrix33) | TypeBit(DataType::Matrix44);
// The data types a mesh draws as a displayed node's result.
inline constexpr DataTypes ResultTypes = TypeBit(DataType::Geometry) | TypeBit(DataType::Instances);

// The attribute type that stores values of a numeric data type.
constexpr AttributeType StorageOf(DataType type) {
    switch (type) {
        case DataType::Float: return AttributeType::Float;
        case DataType::Vec2: return AttributeType::Vec2;
        case DataType::Vec3:
        case DataType::Color3: return AttributeType::Vec3;
        case DataType::Vec4:
        case DataType::Color4: return AttributeType::Vec4;
        case DataType::Int: return AttributeType::Int;
        case DataType::Bool: return AttributeType::Bool;
        case DataType::Matrix33:
        case DataType::Matrix44:
        case DataType::Geometry:
        case DataType::Instances:
        case DataType::Image:
        case DataType::Reference: break;
    }
    std::unreachable();
}
// The data type of stored attribute values.
constexpr DataType DataTypeOf(AttributeType type) {
    switch (type) {
        case AttributeType::Float: return DataType::Float;
        case AttributeType::Vec2: return DataType::Vec2;
        case AttributeType::Vec3: return DataType::Vec3;
        case AttributeType::Vec4: return DataType::Vec4;
        case AttributeType::Int: return DataType::Int;
        case AttributeType::Bool: return DataType::Bool;
    }
    std::unreachable();
}

// A subgraph node's own state for a port of its subgraph: the link into an input port, and what an unlinked input port reads in place of its default.
struct SubgraphNodePort {
    uint16_t PortId{};
    SocketLink Source{};
    std::optional<NumericValue> Value{};
    // The selection an unlinked selection port passes in place of each reader's inline selection.
    std::optional<InlineSelection> Selection{};
};
// Persistent on a node of a geometry or material graph: the subgraph it evaluates, and its own state of each port that has any, sorted by port id.
// The subgraph holds a chain of nodes.
struct SubgraphNode {
    SubgraphId SubgraphId{state::Null};
    std::vector<SubgraphNodePort> Ports;
};
// The user subgraph the id identifies, or null for a built-in subgraph.
inline state::Entity UserSubgraph(const SubgraphId &id) {
    const auto *entity = std::get_if<state::Entity>(&id);
    return entity ? *entity : state::Null;
}

// Evaluates the MaterialX node `Node` with the declaration its input types resolve to.
// Output is the stored output type of a node whose input types do not determine it.
// Literals holds the type of each value input set to a literal, whose value is in its graph's op value block.
// Texts holds each string input that is set.
// Images holds the document texture each unlinked image input reads, by input name.
struct Op {
    std::string Node;
    std::optional<mx::Type> Output;
    std::map<std::string, mx::Type, std::less<>> Literals;
    std::map<std::string, std::string, std::less<>> Texts;
    std::map<std::string, state::Entity, std::less<>> Images;
};

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
    StoreNamedAttribute, StoreSelection, Bake, SetPosition,
    MeshToPoints, InstanceOnPoints, RealizeInstances, ObjectInfo, SubgraphNode, Op, Surface>;
template<typename K>
concept Operated = requires { requires std::is_enum_v<decltype(K::Operation)>; };

enum class SocketDirection : uint8_t { In,
                                       Out };

// A parameter socket names its field by declaration index, and a nested field's member by both indices.
inline constexpr uint8_t NoField = UINT8_MAX;
inline constexpr size_t MaxParameterFields = 16u;
struct Socket {
    DataTypes Types;
    SocketDirection Direction;
    // The value differs across a domain, one per element.
    // An output with it produces such a value, and an input with it accepts one.
    bool Varies{};
    bool Reads{}; // The node reads this geometry input without editing it
    const char *Name{}; // The editor label, or a label from the socket's types when null
    uint8_t Field{NoField};
    uint8_t Member{NoField};
    bool Multi{}; // The input accepts any number of links, kept in order
    bool Selection{}; // The input is a boolean field whose unlinked value is the node's inline selection
    bool Called{}; // The node's position edit evaluates the program a varying input reads at each vertex it moves
    constexpr bool Carries(uint8_t field, uint8_t member) const { return Field == field && Member == member; }
    bool operator==(const Socket &) const = default;
};

namespace detail {
using enum SocketDirection;
inline constexpr bool Single{false}, Varying{true};

inline constexpr DataTypes Bool{TypeBit(DataType::Bool)}, Int{TypeBit(DataType::Int)}, Float{TypeBit(DataType::Float)}, Vec3{TypeBit(DataType::Vec3)}, Vec4{TypeBit(DataType::Vec4)};
inline constexpr DataTypes Geometry{TypeBit(DataType::Geometry)}, Instances{TypeBit(DataType::Instances)}, Image{TypeBit(DataType::Image)};
// The types an image holds, which image nodes read.
inline constexpr DataTypes ImageValues = Float | TypeBit(DataType::Vec2) | Vec3 | Vec4 | TypeBit(DataType::Color3) | TypeBit(DataType::Color4);

constexpr Socket Many(Socket socket) {
    socket.Multi = true;
    return socket;
}
inline constexpr Socket SelectionSocket{.Types = Bool, .Direction = In, .Varies = true, .Selection = true};

inline constexpr Socket SourceSockets[]{{Geometry, Out}};
inline constexpr Socket JoinSockets[]{Many({Geometry, In, Single, true}), {Geometry, Out}};
inline constexpr Socket BooleanSockets[]{{Geometry, In, Single, true, "Mesh 1"}, Many({Geometry, In, Single, true, "Mesh 2"}), {Geometry, Out}};
inline constexpr Socket MinkowskiSockets[]{{Geometry, In, Single, true, "Mesh 1"}, {Geometry, In, Single, true, "Mesh 2"}, {Geometry, Out}};
inline constexpr Socket OperatorSockets[]{{Geometry, In}, SelectionSocket, {Geometry, Out}};
inline constexpr Socket LevelSetSockets[]{{Float, In, Varying, false, "Value"}, {Geometry, Out}};
inline constexpr Socket StoreSockets[]{{Geometry, In}, {NumericTypes, In, Varying, false, "Value"}, {Geometry, Out}};
inline constexpr Socket BakeSockets[]{{Geometry, In, Single, true}, {ImageValues, In, Varying, false, "Value"}, {Image, Out}};
inline constexpr Socket SetPositionSockets[]{{Geometry, In}, {.Types = Vec3, .Direction = In, .Varies = Varying, .Name = "Position", .Called = true}, SelectionSocket, {Geometry, Out}};
inline constexpr Socket PointsSockets[]{{Geometry, In}, {Geometry, Out, Single, false, "Points"}};
inline constexpr Socket WholeSockets[]{{Geometry, In}, {Geometry, Out}};
inline constexpr Socket InstanceSockets[]{{Geometry, In, Single, false, "Points"}, {Geometry, In, Single, true, "Instance"}, {Instances, Out}};
inline constexpr Socket RealizeSockets[]{{Instances, In, Single, true}, {Geometry, Out}};
// A color takes a single number for each component, and a normal takes a color's components.
inline constexpr DataTypes Color3{Float | Vec3 | TypeBit(DataType::Color3)}, Normal3{Vec3 | TypeBit(DataType::Color3)};
inline constexpr Socket SurfaceSockets[]{
    {Color3, In, Varying, false, "base_color"},
    {Float, In, Varying, false, "metallic"},
    {Float, In, Varying, false, "roughness"},
    {Normal3, In, Varying, false, "normal"},
    {Float, In, Varying, false, "occlusion"},
    {Float, In, Varying, false, "specular"},
    {Color3, In, Varying, false, "specular_color"},
    {Float, In, Varying, false, "iridescence"},
    {Float, In, Varying, false, "iridescence_thickness"},
    {Color3, In, Varying, false, "sheen_color"},
    {Float, In, Varying, false, "sheen_roughness"},
    {Float, In, Varying, false, "clearcoat"},
    {Float, In, Varying, false, "clearcoat_roughness"},
    {Normal3, In, Varying, false, "clearcoat_normal"},
    {Color3, In, Varying, false, "emissive"},
    {Float, In, Varying, false, "thickness"},
    {Float, In, Varying, false, "anisotropy_strength"},
    {Float, In, Varying, false, "anisotropy_rotation"},
};

} // namespace detail

// The groups the editor files node kinds under.
enum class KindGroup : uint8_t { Sources,
                                 Operators,
                                 Combine,
                                 Attributes,
                                 Math,
                                 Generators,
                                 Instances };
// What a node kind is apart from its parameters: its fixed sockets, the group the editor files it under, the elements its operator acts on, whether a node is added by its kind, and its help.
// The elements an operator acts on are the domain its linked selection evaluates on without a selection of its own.
struct KindSpec {
    std::span<const Socket> Sockets{detail::OperatorSockets};
    KindGroup Group{KindGroup::Operators};
    Element Natural{Element::Vertex};
    bool Added{true};
    const char *Help{};
};
template<typename K> inline constexpr KindSpec Kind{};
template<> inline constexpr KindSpec Kind<Source>{.Sockets = detail::SourceSockets, .Group = KindGroup::Sources, .Help = "Emits the mesh's source geometry"};
template<> inline constexpr KindSpec Kind<Join>{.Sockets = detail::JoinSockets, .Group = KindGroup::Combine, .Help = "Joins every linked geometry into one mesh, in link order"};
template<> inline constexpr KindSpec Kind<Boolean>{.Sockets = detail::BooleanSockets, .Group = KindGroup::Combine, .Help = "Combines Mesh 1 with each Mesh 2 in turn by union, difference, or intersection. The inputs are closed meshes or regions on one plane."};
template<> inline constexpr KindSpec Kind<Minkowski>{.Sockets = detail::MinkowskiSockets, .Group = KindGroup::Combine, .Help = "Sweeps one closed mesh or region by another, dilating or eroding it"};
template<> inline constexpr KindSpec Kind<primitive::Plane>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a grid of quads in the XZ plane, facing +Y"};
template<> inline constexpr KindSpec Kind<primitive::Circle>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a circle in the XZ plane, facing +Y"};
template<> inline constexpr KindSpec Kind<primitive::Cube>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a box centered on the origin"};
template<> inline constexpr KindSpec Kind<primitive::IcoSphere>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a sphere by subdividing an icosahedron"};
template<> inline constexpr KindSpec Kind<primitive::UVSphere>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a sphere of slices around the Y axis and stacks along it"};
template<> inline constexpr KindSpec Kind<primitive::QuadSphere>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a sphere of quads by mapping a subdivided cube onto it"};
template<> inline constexpr KindSpec Kind<primitive::Torus>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a torus around the Y axis"};
template<> inline constexpr KindSpec Kind<primitive::Cylinder>{.Sockets = detail::SourceSockets, .Group = KindGroup::Generators, .Help = "Generates a cylinder, frustum or cone along the Y axis, centered on the origin"};
template<> inline constexpr KindSpec Kind<LevelSet>{.Sockets = detail::LevelSetSockets, .Group = KindGroup::Generators, .Help = "Marches the surface where a field crosses a level over a box"};
template<> inline constexpr KindSpec Kind<action::mesh::Delete>{.Help = "Deletes the selected vertices, edges, or faces"};
template<> inline constexpr KindSpec Kind<action::mesh::Merge>{.Help = "Merges the selected vertices into one, or merges nearby pairs"};
template<> inline constexpr KindSpec Kind<Extrude>{.Help = "Extrudes the selected elements in steps, rotating about an axis and moving by the offset, joined to the geometry they came from"};
template<> inline constexpr KindSpec Kind<action::mesh::Duplicate>{.Help = "Duplicates the selected faces in place onto copied vertices"};
template<> inline constexpr KindSpec Kind<action::mesh::Split>{.Help = "Disconnects the selected faces from the unselected geometry around them"};
template<> inline constexpr KindSpec Kind<action::mesh::Dissolve>{.Help = "Removes the selected elements and joins the faces around them"};
template<> inline constexpr KindSpec Kind<action::mesh::Subdivide>{.Natural = Element::Edge, .Help = "Cuts the selected edges and splits their faces into smaller faces"};
template<> inline constexpr KindSpec Kind<SubdivisionSurface>{.Sockets = detail::WholeSockets, .Help = "Smooths the whole mesh by Catmull-Clark subdivision"};
template<> inline constexpr KindSpec Kind<SmoothSurface>{.Sockets = detail::WholeSockets, .Help = "Refines a closed mesh onto smooth patches through its shading normals, so flat-shaded faces and sharp edges stay creased"};
template<> inline constexpr KindSpec Kind<action::mesh::SplitFaces>{.Natural = Element::Face, .Help = "Splits the selected faces until each part is a triangle, is convex, or is planar within Angle"};
template<> inline constexpr KindSpec Kind<action::mesh::TrisToQuads>{.Natural = Element::Face, .Help = "Joins pairs of selected triangles into quads"};
template<> inline constexpr KindSpec Kind<action::mesh::Poke>{.Natural = Element::Face, .Help = "Fans each selected face around a new center vertex"};
template<> inline constexpr KindSpec Kind<action::mesh::FlipNormals>{.Natural = Element::Face, .Help = "Reverses the winding and normals of the selected faces"};
template<> inline constexpr KindSpec Kind<action::mesh::EdgeSplit>{.Natural = Element::Edge, .Help = "Splits the mesh along the selected edges, disconnecting the faces on either side"};
template<> inline constexpr KindSpec Kind<action::mesh::Inset>{.Natural = Element::Face, .Help = "Insets the selected faces, as one region or each face on its own"};
template<> inline constexpr KindSpec Kind<action::mesh::Solidify>{.Natural = Element::Face, .Help = "Thickens the selected faces into a shell, with a rim along the region boundary"};
template<> inline constexpr KindSpec Kind<action::mesh::Wireframe>{.Natural = Element::Face, .Help = "Builds solid struts along the edges of the selected faces"};
template<> inline constexpr KindSpec Kind<action::mesh::Bevel>{.Help = "Bevels the selected edges or vertices"};
template<> inline constexpr KindSpec Kind<action::mesh::SmoothVertices>{.Help = "Moves the selected vertices toward the mean of their edge neighbors"};
template<> inline constexpr KindSpec Kind<action::mesh::ShrinkFatten>{.Help = "Moves the selected vertices along their normals"};
template<> inline constexpr KindSpec Kind<action::mesh::Randomize>{.Help = "Offsets the selected vertices by repeatable random amounts"};
template<> inline constexpr KindSpec Kind<action::mesh::SpaceEvenly>{.Natural = Element::Edge, .Help = "Spaces the vertices of the selected edge chains evenly along them"};
template<> inline constexpr KindSpec Kind<action::mesh::RelaxEdgeLoops>{.Natural = Element::Edge, .Help = "Smooths the selected edge chains, keeping their open endpoints and junctions"};
template<> inline constexpr KindSpec Kind<action::mesh::CurveBetweenSelected>{.Help = "Moves the unselected loop vertices onto a curve through the selected ones"};
template<> inline constexpr KindSpec Kind<action::mesh::Circularize>{.Natural = Element::Edge, .Help = "Fits the selected boundaries and wire chains to circles"};
template<> inline constexpr KindSpec Kind<action::mesh::MakePlanarFaces>{.Natural = Element::Face, .Help = "Flattens the selected faces toward their planes"};
template<> inline constexpr KindSpec Kind<Flatten>{.Help = "Flattens the selected face regions and edge chains onto planes"};
template<> inline constexpr KindSpec Kind<ToSphere>{.Help = "Blends the selected vertices toward a sphere around Center"};
template<> inline constexpr KindSpec Kind<PushPull>{.Help = "Moves the selected vertices toward or away from Center"};
template<> inline constexpr KindSpec Kind<Shear>{.Help = "Shears the selected vertices along Direction"};
template<> inline constexpr KindSpec Kind<Warp>{.Help = "Warps the selected vertices around Center"};
template<> inline constexpr KindSpec Kind<Bend>{.Help = "Bends the selected vertices from Center"};
template<> inline constexpr KindSpec Kind<VertexSlide>{.Help = "Slides each selected vertex along its incident edge nearest Direction"};
template<> inline constexpr KindSpec Kind<EdgeSlide>{.Natural = Element::Edge, .Help = "Slides the selected edge chains along their adjacent faces"};
template<> inline constexpr KindSpec Kind<action::mesh::SnapSymmetry>{.Help = "Snaps the selected vertices to their mirrored partners across an axis plane"};
template<> inline constexpr KindSpec Kind<Bisect>{.Help = "Cuts the mesh along a plane and deletes the vertices beyond it on the cleared sides"};
template<> inline constexpr KindSpec Kind<Project>{.Sockets = detail::WholeSockets, .Help = "Leaves the outline of the mesh seen along a direction, as a region on a plane"};
template<> inline constexpr KindSpec Kind<action::mesh::ConvexHull>{.Help = "Replaces the selection with the convex hull of the selected vertices"};
template<> inline constexpr KindSpec Kind<action::mesh::Simplify>{.Help = "Collapses edges within a tolerance, then toward a triangle ratio"};
template<> inline constexpr KindSpec Kind<action::mesh::RecalculateNormals>{.Natural = Element::Face, .Help = "Makes the selected faces' normals point outside each closed surface they belong to"};
template<> inline constexpr KindSpec Kind<action::mesh::ConnectVertices>{.Help = "Splits faces along chords between their consecutive selected vertices"};
template<> inline constexpr KindSpec Kind<action::mesh::Fill>{.Natural = Element::Edge, .Help = "Creates an edge from two selected vertices, fills the selected loops of at most Sides edges, or forms a face from the selected points"};
template<> inline constexpr KindSpec Kind<action::mesh::BridgeEdgeLoops>{.Natural = Element::Edge, .Help = "Bridges two selected edge loops or chains with a strip of faces"};
template<> inline constexpr KindSpec Kind<action::mesh::GridFill>{.Natural = Element::Edge, .Help = "Fills one selected closed loop with a grid of quads"};
template<> inline constexpr KindSpec Kind<action::mesh::EdgeRotate>{.Natural = Element::Edge, .Help = "Rotates each selected edge between its two faces"};
template<> inline constexpr KindSpec Kind<action::mesh::BeautifyFaces>{.Natural = Element::Face, .Help = "Rearranges the selected triangles into more evenly shaped ones"};
template<> inline constexpr KindSpec Kind<action::mesh::Unsubdivide>{.Help = "Reverses subdivision by removing alternate selected vertices of regular regions"};
template<> inline constexpr KindSpec Kind<Knife>{.Sockets = detail::WholeSockets, .Help = "Cuts every edge whose projection in a view crosses a screen segment"};
template<> inline constexpr KindSpec Kind<action::mesh::CycleCornerAttribute>{.Natural = Element::Face, .Help = "Shifts or reverses the UVs or colors around the corners of the selected faces"};
template<> inline constexpr KindSpec Kind<MarkSharp>{.Natural = Element::Edge, .Help = "Marks the selected edges sharp or smooth, or shades the selected faces flat or smooth"};
template<> inline constexpr KindSpec Kind<Shade>{.Sockets = detail::WholeSockets, .Help = "Shades every face flat, smooth, or smooth with sharp edges where adjacent faces meet at more than Angle"};
template<> inline constexpr KindSpec Kind<TransformElements>{.Help = "Moves, rotates, and scales the selected vertices, all by default, around a pivot, after reflecting them across a mirror plane through it"};
template<> inline constexpr KindSpec Kind<StoreNamedAttribute>{.Sockets = detail::StoreSockets, .Group = KindGroup::Attributes, .Help = "Writes the Value input to the named attribute of the geometry"};
template<> inline constexpr KindSpec Kind<StoreSelection>{.Group = KindGroup::Attributes, .Help = "Writes the selection to the named Boolean attribute of the geometry"};
template<> inline constexpr KindSpec Kind<Bake>{.Sockets = detail::BakeSockets, .Group = KindGroup::Attributes, .Help = "Stores the Value field over the geometry's UV layout in an image"};
template<> inline constexpr KindSpec Kind<SetPosition>{.Sockets = detail::SetPositionSockets, .Group = KindGroup::Attributes, .Help = "Sets or offsets the selected vertices' positions from a vector attribute"};
template<> inline constexpr KindSpec Kind<MeshToPoints>{.Sockets = detail::PointsSockets, .Group = KindGroup::Instances, .Help = "Emits the input's vertices as points"};
template<> inline constexpr KindSpec Kind<InstanceOnPoints>{.Sockets = detail::InstanceSockets, .Group = KindGroup::Instances, .Help = "Places an instance of the second input at each point of the first"};
template<> inline constexpr KindSpec Kind<RealizeInstances>{.Sockets = detail::RealizeSockets, .Group = KindGroup::Instances, .Help = "Emits a copy of the instance geometry for each instance, placed by its transform"};
template<> inline constexpr KindSpec Kind<ObjectInfo>{.Sockets = detail::SourceSockets, .Group = KindGroup::Sources, .Help = "Emits another object's mesh in that mesh's local space"};
// A subgraph node's and an op node's sockets come from their subgraphs.
template<> inline constexpr KindSpec Kind<SubgraphNode>{.Sockets = {}, .Help = "Uses a subgraph, whose ports are its inputs and outputs"};
template<> inline constexpr KindSpec Kind<Op>{.Sockets = {}, .Group = KindGroup::Math, .Added = false, .Help = "Evaluates a MaterialX node on its inputs"};
template<> inline constexpr KindSpec Kind<Surface>{.Sockets = detail::SurfaceSockets, .Added = false, .Help = "Computes the material's linked inputs at each surface point, and the material's own values set the rest, alpha and transmission always, since they decide visibility before the graph runs"};

namespace detail {
template<typename F> constexpr DataTypes TypesOf() {
    if constexpr (std::same_as<F, float>) return Float;
    else if constexpr (std::same_as<F, uint32_t>) return Int;
    else if constexpr (std::same_as<F, bool>) return Bool;
    else if constexpr (std::same_as<F, vec3>) return Vec3;
    else if constexpr (std::same_as<F, quat>) return Vec4;
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

// Per-element parameters have varying sockets, and other numeric parameters do not vary.
template<typename K> inline constexpr field::Table<K, DataTypes> PerElement{};
template<> inline constexpr field::Table<Extrude, DataTypes> PerElement<Extrude>{
    {&Extrude::Offset, Vec3},
    {&Extrude::OffsetScale, Float},
};
template<> inline constexpr field::Table<action::mesh::Bevel, DataTypes> PerElement<action::mesh::Bevel>{{&action::mesh::Bevel::Width, Float}};
template<> inline constexpr field::Table<action::mesh::Poke, DataTypes> PerElement<action::mesh::Poke>{{&action::mesh::Poke::Offset, Float}};
template<> inline constexpr field::Table<action::mesh::Solidify, DataTypes> PerElement<action::mesh::Solidify>{{&action::mesh::Solidify::Thickness, Float}};
template<> inline constexpr field::Table<action::mesh::Wireframe, DataTypes> PerElement<action::mesh::Wireframe>{{&action::mesh::Wireframe::Thickness, Float}};
template<> inline constexpr field::Table<action::mesh::Inset, DataTypes> PerElement<action::mesh::Inset>{
    {&action::mesh::Inset::Thickness, Float},
    {&action::mesh::Inset::Depth, Float},
};
template<> inline constexpr field::Table<action::mesh::ShrinkFatten, DataTypes> PerElement<action::mesh::ShrinkFatten>{{&action::mesh::ShrinkFatten::Distance, Float}};
template<> inline constexpr field::Table<action::mesh::SmoothVertices, DataTypes> PerElement<action::mesh::SmoothVertices>{{&action::mesh::SmoothVertices::Factor, Float}};
template<> inline constexpr field::Table<action::mesh::Randomize, DataTypes> PerElement<action::mesh::Randomize>{{&action::mesh::Randomize::Amount, Float}};
// The kinds whose run is a position edit, which evaluates a program a per-element parameter reads at each vertex it moves.
template<typename K> inline constexpr bool EditsPositions{false};
template<> inline constexpr bool EditsPositions<action::mesh::ShrinkFatten>{true};
template<> inline constexpr bool EditsPositions<action::mesh::SmoothVertices>{true};
template<> inline constexpr bool EditsPositions<action::mesh::Randomize>{true};
template<> inline constexpr field::Table<InstanceOnPoints, DataTypes> PerElement<InstanceOnPoints>{
    {&InstanceOnPoints::Scale, Vec3},
    {&InstanceOnPoints::Rotation, Vec4},
};

template<typename K> inline constexpr auto KindSocketTable = [] {
    constexpr auto fixed = Kind<K>.Sockets;
    std::array<Socket, fixed.size() + MaxParameterFields> all{};
    size_t n = 0u;
    for (const auto &socket : fixed) {
        if (socket.Direction == In) all[n++] = socket;
    }
    if constexpr (field::Walkable<K> && !std::same_as<K, Op> && !std::same_as<K, SubgraphNode>) {
        K value{};
        field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
            using F = std::remove_cvref_t<decltype(member)>;
            constexpr auto types = TypesOf<F>();
            if constexpr (types != 0u) {
                constexpr auto per_element = PerElement<K>.ByIndex[I];
                all[n++] = {.Types = per_element ? per_element : types, .Direction = In, .Varies = per_element != 0, .Name = field::Label<K, I>.c_str(), .Field = uint8_t(I), .Called = per_element != 0 && EditsPositions<K>};
            } else if constexpr (field::Walkable<F>) {
                field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                    constexpr auto nested_types = TypesOf<std::remove_cvref_t<decltype(nested)>>();
                    if constexpr (nested_types != 0u) {
                        all[n++] = {nested_types, In, Single, false, NestedLabel<K, I, F, J>.c_str(), uint8_t(I), uint8_t(J)};
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
inline constexpr auto GeneratorKind = detail::PerKind(NodeKinds{}, []<typename K> { return Generator<K>; });

inline constexpr uint16_t KindCount = uint16_t(SocketCount.size());
inline constexpr size_t MaxInputs = std::ranges::max(InputCount);

// Each kind's multi-input socket, or NoSocket.
inline constexpr uint16_t NoSocket = UINT16_MAX;
inline constexpr auto MultiInput = detail::PerKind(NodeKinds{}, []<typename K> {
    constexpr auto sockets = detail::SocketsOf<K>();
    static_assert(std::ranges::count(sockets, true, &Socket::Multi) <= 1, "A node kind has at most one multi-input socket");
    const auto found = std::ranges::find(sockets, true, &Socket::Multi);
    return found == sockets.end() ? NoSocket : uint16_t(found - sockets.begin());
});
// A node has one input slot per input socket, then one slot per extra link of its multi-input socket, in order.
// The first link of a multi-input socket is in the socket's own slot.
// A node whose sockets come from its subgraph has one slot per input socket.
constexpr uint16_t SlotSocket(uint16_t kind, uint16_t slot) { return MultiInput[kind] != NoSocket && slot >= InputCount[kind] ? MultiInput[kind] : slot; }
constexpr uint16_t SlotOf(uint16_t kind, uint16_t socket, uint16_t index) { return index == 0u ? socket : uint16_t(InputCount[kind] + index - 1u); }
constexpr uint16_t NextMultiSlot(uint16_t kind, uint16_t slot) { return slot < InputCount[kind] ? InputCount[kind] : uint16_t(slot + 1u); }

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

// Visits the socket's parameter as (value, spec), where a nested member without bounds or help takes its field's.
constexpr void VisitSocketParameter(auto &parameters, Socket socket, auto &&visit) {
    using P = std::remove_cvref_t<decltype(parameters)>;
    if constexpr (field::Walkable<P>) {
        field::ForEach(parameters, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
            if (socket.Field != I) return;
            using F = std::remove_cvref_t<decltype(member)>;
            if constexpr (detail::TypesOf<F>() != 0u) {
                if (socket.Member == NoField) visit(member, Spec<P, I>);
            } else if constexpr (field::Walkable<F>) {
                field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                    if constexpr (detail::TypesOf<std::remove_cvref_t<decltype(nested)>>() != 0u) {
                        static constexpr auto spec = [] {
                            auto spec = Spec<F, J>;
                            if (!spec.Bounded()) {
                                spec.Min = Spec<P, I>.Min;
                                spec.Max = Spec<P, I>.Max;
                            }
                            if (!spec.Help) spec.Help = Spec<P, I>.Help;
                            return spec;
                        }();
                        if (socket.Member == J) visit(nested, spec);
                    }
                });
            }
        });
    }
}
// The words hold the socket's value, an unsigned field's as a signed MaterialX integer, and the field's range clamps it as a direct edit's.
void AssignSocketParameter(auto &parameters, Socket socket, const AttributeValue &words) {
    VisitSocketParameter(parameters, socket, [&]<typename F>(F &value, const FieldSpec &spec) {
        if constexpr (std::same_as<F, uint32_t>) {
            value = F(std::clamp<int64_t>(std::bit_cast<int32_t>(words[0]), action::LowerBound<F>(spec), action::UpperBound<F>(spec)));
        } else {
            std::memcpy(&value, words.data(), sizeof(value));
            if constexpr (action::ScalarField<F> || action::VectorField<F>) value = action::ClampField(value, spec);
        }
    });
}

constexpr std::pair<uint16_t, uint16_t> ParameterRange(uint16_t kind) {
    return {uint16_t(std::ranges::count(KindSockets(kind).first(InputCount[kind]), NoField, &Socket::Field)), InputCount[kind]};
}

// Whether the editor puts the input socket in the node's tray.
// Parameters that do not vary go in the tray when the kind has more than one.
// Varying parameters, and a kind's only parameter that does not vary, get their own rows.
constexpr bool InTray(uint16_t kind, uint16_t socket) {
    const auto sockets = KindSockets(kind).first(InputCount[kind]);
    const auto setting = [](Socket s) { return s.Field != NoField && !s.Varies; };
    return socket < sockets.size() && setting(sockets[socket]) && std::ranges::count_if(sockets, setting) > 1;
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

// Accept an output whose data types intersect the input's, where a varying output needs an input that accepts a varying value.
// A selection port read inside its subgraph links only to selection inputs, since its unlinked value is each reader's inline selection.
constexpr bool Links(Socket from, Socket to) { return (from.Types & to.Types) && (!from.Varies || to.Varies) && (!from.Selection || to.Selection); }

void VisitKind(uint16_t kind, auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) {
        uint16_t index = 0u;
        (void)((index++ == kind && (f.template operator()<Ks>(), true)) || ...);
    }(NodeKinds{});
}

// Cooks the handle support of the node's occurrence in the subgraph nodes of `path` of the mesh's graph, and returns whether it has one.
bool PrepareOperationHandle(state::Registry &, state::Entity mesh_entity, std::span<const state::Entity> path, state::Entity node);
// A new record holding the geometry an input slot of the node's occurrence in the subgraph nodes of `path` of the mesh's graph reads.
uint32_t SelectionInput(state::Registry &, state::Entity mesh_entity, std::span<const state::Entity> path, state::Entity node, uint16_t slot);
void SetMeshSelection(state::Registry &, state::Entity mesh, const InlineSelection *);
// The elements the node's operator acts on, which its linked selection evaluates on without a selection of its own.
Element NaturalDomainOf(const state::Registry &, state::Entity node);

void ForEachKind(auto &&f) {
    [&]<typename... Ks>(TypeList<Ks...>) { (f.template operator()<Ks>(), ...); }(NodeKinds{});
}

inline constexpr uint32_t NoNode = UINT32_MAX;

// Parent is the subgraph node whose subgraph holds the node, or NoNode for a node of the planned graph.
// InBuiltinSubgraph is set for a node of a built-in subgraph, whose entity is in the registry of built-in subgraphs.
struct NodeRecord {
    state::Entity Entity;
    uint16_t Kind;
    uint32_t Parent{NoNode};
    bool InBuiltinSubgraph{};
    bool operator==(const NodeRecord &) const = default;
};

// Next is the node after this one in authoring order, or null for the last.
// Graph is the mesh entity of the geometry graph or the entity of the subgraph whose graph holds the node.
struct GraphNode {
    state::Entity Graph{state::Null};
    state::Entity Next{state::Null};
    uint16_t Kind{};
};
// A node occurrence in a mesh's cook: the subgraph nodes enclosing it, outermost first, then the node.
using NodePath = std::vector<state::Entity>;

// A port has the data types and the selection property of the socket it was made from.
// A geometry subgraph's output port has the source inside its graph, and its input port has none.
// An input port without a default reads what the unlinked socket it exposes reads.
struct SubgraphPort {
    uint16_t Id{};
    std::string Name;
    DataTypes Types{};
    bool Selection{};
    SocketDirection Direction{SocketDirection::In};
    PortBinding Binding;
    std::optional<NumericValue> Default;
    SocketLink &Source() { return std::get<SocketLink>(Binding); }
    const SocketLink &Source() const { return std::get<SocketLink>(Binding); }
};
// Whether the port is a numeric input other than a selection, which reads a value while unlinked.
inline bool ReadsDefault(const SubgraphPort &port) { return port.Direction == SocketDirection::In && (port.Types & NumericTypes) && !port.Selection; }
struct SubgraphInterface {
    uint32_t NextId{};
    std::vector<SubgraphPort> Ports;
};
// Persistent on a node graph: a geometry graph's mesh entity, a subgraph's entity or a material whose inputs a node graph computes.
// The first and last node of the graph in authoring order.
struct GraphNodes {
    state::Entity First{state::Null}, Last{state::Null};
    bool operator==(const GraphNodes &) const = default;
};
// Persistent on every subgraph's entity, which also holds its SubgraphInterface and Name.
// A subgraph of a geometry or material graph holds its nodes as a chain in GraphNodes.
// A subgraph of a scene's or the document's graph holds its items in SubgraphItems.
struct Subgraph {};
// Persistent on a material entity whose inputs a node graph computes, which also holds its GraphNodes.
// The graph holds one Surface node.
struct MaterialGraph {};
// The registry holding a subgraph and its entity there.
struct SubgraphRef {
    const state::Registry *Registry{};
    state::Entity Entity{state::Null};
};
// The built-in subgraphs, which a registry of their own holds once per session.
const state::Registry &BuiltinSubgraphRegistry();
SubgraphRef BuiltinSubgraph(BuiltinSubgraphId);
// The subgraph the subgraph node evaluates.
SubgraphRef SubgraphOf(const state::Registry &, state::Entity node);
// The registry holding the graph at the end of the path of subgraph nodes, which is the registry of built-in subgraphs once a subgraph node of the path uses one.
const state::Registry &RegistryAt(const state::Registry &, std::span<const state::Entity> path);
// The first node of a geometry graph or a subgraph.
state::Entity FirstNode(const state::Registry &, state::Entity graph);
std::vector<state::Entity> NodesOf(const state::Registry &, state::Entity graph);
// Whether the graph is the user subgraph or uses it, at any depth.
bool Uses(const state::Registry &, state::Entity graph, state::Entity subgraph);
// Each user subgraph with each mesh whose graph uses it at any depth, ordered by subgraph.
std::vector<std::pair<state::Entity, state::Entity>> SubgraphUses(const state::Registry &);
// The meshes whose graphs hold the graph: a geometry graph's mesh itself, or each mesh that uses the subgraph among `uses`.
std::vector<state::Entity> MeshesUsing(const state::Registry &, std::span<const std::pair<state::Entity, state::Entity>> uses, state::Entity graph);
// The meshes whose graphs hold the graph, which for a user subgraph are those SubgraphUses gives.
std::vector<state::Entity> MeshesUsing(const state::Registry &, state::Entity graph);
// The subgraph node's own state of the port, or null.
const SubgraphNodePort *NodePortOf(const state::Registry &, state::Entity node, uint16_t port);
// The subgraph node's name, which is its subgraph's.
std::string SubgraphNodeName(const state::Registry &, state::Entity node);
// Appends the node to the graph's authoring order with the input links.
void AddToGraph(state::Registry &, state::Entity graph, NodeRecord, std::vector<SocketLink> links);
// Makes the nodes the graph's nodes in this authoring order.
void OrderGraph(state::Registry &, state::Entity graph, std::span<const state::Entity> nodes);
// A new empty user subgraph with a unique name.
state::Entity CreateSubgraph(state::Registry &, std::string_view name);
state::Entity CreateSubgraphNode(state::Registry &, state::Entity graph, SubgraphId);
// Whether the entity is a geometry graph or a material whose inputs a node graph computes, whose graph node actions edit.
bool IsNodeGraph(const state::Registry &, state::Entity);
state::Entity SurfaceNodeOf(const state::Registry &, state::Entity material);
// Each source node and its copy, by source.
struct NodeCopies {
    std::vector<std::pair<state::Entity, state::Entity>> Items;
    state::Entity operator()(state::Entity source) const;
};
// Maps a link of a copied node to a node that is not copied.
using LinkMap = std::function<SocketLink(SocketLink)>;
// Creates a copy of each node of `from` with each of its components but its place in a graph and its input links.
// A copied subgraph node's port links read the copies, or else what `link` maps them to.
NodeCopies CopyNodes(state::Registry &, const state::Registry &from, std::span<const state::Entity> originals, const LinkMap &link);
// The input links of the node of `from` that its copy reads: the copies, or else what `link` maps them to.
std::vector<SocketLink> CopiedInputs(const state::Registry &from, state::Entity node, const NodeCopies &, const LinkMap &link);
// Appends a copy of each node of `from` to the graph in order, as CopyNodes copies them, and lays out the graph's values with each copied op's literals.
NodeCopies CopyInto(state::Registry &, state::Entity graph, const state::Registry &from, std::span<const state::Entity> originals, const LinkMap &link);
// Rewrites each node occurrence every mesh holds: the occurrences its cooks kept, its record's tip and its displayed node.
void RewritePaths(state::Registry &, const std::function<void(NodePath &)> &);

struct SocketEdit {
    state::Entity Node;
    uint16_t Socket;
    SocketLink Source;
};
// A link resolved to the output socket that produces its value, in the node occurrence Path, held by Registry.
// An unlinked numeric input port gives a constant, and an unlinked selection port of a subgraph node with a selection of its own gives that subgraph node and port.
struct ResolvedSocket {
    NodePath Path;
    const state::Registry *Registry{};
    SocketLink Source;
    std::optional<NumericValue> Constant;
    state::Entity SelectionUse{state::Null};
    uint16_t SelectionPort{};
    bool Cycle{};
    // The subgraph node whose unlinked input port's value or default Constant is, under Path, and the port.
    state::Entity ValueUse{state::Null};
    uint16_t ValuePort{};
};
// The graph whose page shows the node.
state::Entity GraphOf(const state::Registry &, state::Entity node);
// The port of a subgraph node's subgraph, a subgraph's own port, or a scene subgraph's port.
const SubgraphPort *PortOf(const state::Registry &, state::Entity use, uint16_t port);
const SubgraphInterface *InterfaceOf(const state::Registry &, state::Entity use);
// The port as a socket facing the direction: facing out, it passes its resolved source's value, and facing in, it accepts any value.
// A subgraph's input port read inside its graph passes a single value, and each link into a subgraph node's input port is checked against the inputs it reaches.
Socket PortSocket(const state::Registry &, state::Entity use, const SubgraphPort &, SocketDirection facing);
Socket SocketOf(const state::Registry &, state::Entity node, uint16_t socket);
// The sockets of a node other than a subgraph: its kind's, or an op node's from its subgraph, where no output varies.
std::span<const Socket> SocketsOf(const state::Registry &, state::Entity node);
uint16_t InputCountOf(const state::Registry &, state::Entity node);
// The socket of a node's input slot, which for a slot past its input sockets is its multi-input socket, and for a subgraph node is the port of the slot's id.
uint16_t SlotSocketOf(const state::Registry &, state::Entity node, uint16_t slot);
// Calls `f(slot, link)` for each input link the node holds when the call starts, so `f` may set the node's inputs: a subgraph node's port links by port id, or another node's links by slot.
void ForEachInput(const state::Registry &, state::Entity node, const std::function<void(uint16_t slot, SocketLink)> &f);
// Calls `f(node, slot, link)` for each input link of each node of the graph, then for the source of each output port of a subgraph.
void ForEachGraphInput(const state::Registry &, state::Entity graph, const std::function<void(state::Entity node, uint16_t slot, SocketLink)> &f);
// The link in a node's input slot, a subgraph node's input port's link, or a subgraph's output port's source.
SocketLink InputOf(const state::Registry &, state::Entity node, uint16_t slot);
// A null link in a multi-input socket's slot removes that link from the socket's list.
void SetInput(state::Registry &, state::Entity node, uint16_t slot, SocketLink);
// Inserts the link at `index` among the links of the node's multi-input socket, or at the end.
// If the socket already has the source, that link moves to `index`.
void InsertInput(state::Registry &, state::Entity node, SocketLink, uint16_t index);
std::vector<SocketLink> InputLinks(const state::Registry &, state::Entity node, uint16_t socket);
// Resolves a link read in the graph of the last subgraph node of `path`, or of the graph that holds the link's node for an empty path.
ResolvedSocket ResolveSocket(const state::Registry &, NodePath path, SocketLink, bool bypass = false, const SocketEdit * = nullptr);
inline ResolvedSocket ResolveSocket(const state::Registry &r, SocketLink link, bool bypass = false, const SocketEdit *edit = nullptr) { return ResolveSocket(r, NodePath{}, link, bypass, edit); }
std::optional<uint16_t> SubgraphBypassInput(const state::Registry &, state::Entity, uint16_t output);
std::optional<NumericValue> ParameterDefault(const state::Registry &, state::Entity node, uint16_t socket);

// One link per input slot.
// The links of a multi-input socket are distinct, and none is null unless the socket has no links.
struct NodeInputs {
    std::vector<SocketLink> Links;
};
std::vector<SocketLink> NormalizedInputs(uint16_t kind, std::vector<SocketLink>);

// The output socket carrying a node's geometry or instances, which the mesh draws when the node is displayed, or none.
constexpr std::optional<uint16_t> ResultSocket(uint16_t kind) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = InputCount[kind]; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Types & ResultTypes) return s;
    }
    return {};
}

// Persistent on a mesh entity with a node graph, which also holds its GraphNodes.
// DisplayNode is in the graph of the last subgraph node of DisplayPath, whose first subgraph node is in the graph, or in the graph for an empty path.
struct GeometryGraph {
    state::Entity DisplayNode{state::Null};
    std::optional<uint16_t> DisplayPort;
    std::vector<state::Entity> DisplayPath;
    bool operator==(const GeometryGraph &) const = default;
};

std::optional<uint16_t> ResultSocket(const state::Registry &, state::Entity);
SocketLink DisplayLink(const state::Registry &, const GeometryGraph &);
// The displayed node's occurrence.
NodePath DisplayedPath(const GeometryGraph &);

struct NodeBypassed {};

// The output socket an input reads by node index in its plan, or NoNode when unlinked.
struct NodeLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
    bool operator==(const NodeLink &) const = default;
};
// A link that reads an unlinked selection port of a subgraph node with a selection of its own has that subgraph node as SelectionUse.
struct PlanLink {
    uint32_t FromNode{NoNode};
    uint16_t FromSocket{};
    std::optional<NumericValue> Constant;
    uint32_t SelectionUse{NoNode};
    uint16_t SelectionPort{};
    bool operator==(const PlanLink &) const = default;
};
// What a plan reads of a subgraph node of its own graph other than its port values: its place, its bypass, its subgraph, and its ports' links and which of them hold a value or a selection.
struct SubgraphShape {
    state::Entity Graph, Next;
    SubgraphId SubgraphId;
    bool Bypassed;
    std::vector<std::tuple<uint16_t, SocketLink, bool, bool>> Ports;
    bool operator==(const SubgraphShape &) const = default;
};
// An input of a plan that reads the value or default of an unlinked input port of the subgraph node occurrence Use.
struct PortRead {
    uint32_t Input, Use;
    uint16_t Port;
};
// A graph's nodes with each subgraph node followed by the nodes of its subgraph, at any depth.
// A subgraph node has no sockets in a compiled plan, and inputs resolve through its ports.
struct CookPlan {
    uint64_t Revision{};
    std::vector<Socket> ViewSockets; // Each node's sockets from its SocketFirst, where an op output varies as its links make it
    std::vector<NodeRecord> Nodes; // In authoring order
    std::vector<uint32_t> ByEntity; // Node indices by parent, then entity
    std::vector<state::Entity> Paths; // Each node's occurrence from its PathFirst
    std::vector<uint32_t> PathFirst;
    std::vector<PlanLink> Inputs; // Resolved execution inputs
    uint32_t DisplayNode{NoNode}, ResultNode{NoNode};
    std::vector<uint32_t> Order; // Every node after the nodes it reads
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input and first socket slot, then the totals
    std::vector<uint32_t> Consumers; // The inputs reading each socket slot
    std::vector<bool> Demanded; // Resolved ancestors of the displayed result
    std::vector<state::Entity> Subgraphs; // The user subgraphs the plan's subgraph nodes use, at any depth, sorted
    std::vector<PortRead> PortReads;
    std::vector<std::pair<uint32_t, SubgraphShape>> SubgraphNodes; // Each subgraph node of the plan's own graph and its shape
};
// The plan nodes of the demanded op occurrences that read the timeline's frame, and those that read its time, on the mesh entity of a graph with any.
struct TimeReads {
    std::vector<uint32_t> Frame, Time;
};
// The generator occurrences that write texture coordinate set 0, on the mesh entity of a graph whose plan has a generator.
// Every one writes them where All holds, which a material the mesh draws that reads the set or another graph that reads the mesh makes hold.
// Otherwise each one in Occurrences writes them, which a demanded node after it reads the set through.
struct TexcoordWriters {
    bool All;
    std::vector<NodePath> Occurrences;
};
// The registry holding the plan node's entity.
inline const state::Registry &Authored(const state::Registry &r, const NodeRecord &node) { return node.InBuiltinSubgraph ? BuiltinSubgraphRegistry() : r; }
// Plans a geometry graph, or a subgraph whose input ports read nothing.
// The plan's display and result nodes stay unbound, which Compile binds.
CookPlan Plan(const state::Registry &, state::Entity graph, bool bypass = false, const SocketEdit * = nullptr);
CookPlan Compile(const state::Registry &, state::Entity graph);
// The compiled plan with the port values the `changed` subgraph nodes of its own graph now hold, or none when any of them changed what else the plan read of it.
std::optional<CookPlan> Revalued(const state::Registry &, const CookPlan &, std::span<const state::Entity> changed);
bool ObjectReferencesAcyclic(const state::Registry &, std::span<const state::Entity> nodes, state::Entity object, state::Entity additional_reader = state::Null, std::optional<std::pair<state::Entity, state::Entity>> binding = {});
void BindDisplay(const state::Registry &, const GeometryGraph &, CookPlan &);
// Demands the nodes and the nodes they read, and counts the demanded readers of each socket.
void Demand(CookPlan &, std::span<const uint32_t> nodes);
inline uint16_t InputsOf(const CookPlan &plan, uint32_t node) { return uint16_t(plan.InputFirst[node + 1] - plan.InputFirst[node]); }
inline std::span<const Socket> NodeSockets(const CookPlan &plan, uint32_t node) {
    return plan.ViewSockets.empty() ? KindSockets(plan.Nodes[node].Kind) : std::span{plan.ViewSockets}.subspan(plan.SocketFirst[node], plan.SocketFirst[node + 1] - plan.SocketFirst[node]);
}
inline std::span<const PlanLink> NodeLinks(const CookPlan &plan, uint32_t node) { return std::span{plan.Inputs}.subspan(plan.InputFirst[node], InputsOf(plan, node)); }
std::optional<uint16_t> BypassInput(uint16_t kind);
// The active node of a geometry graph or a subgraph.
state::Entity ActiveNodeOf(const state::Registry &, state::Entity graph);
// Orders the nodes by parent, entity and index, and lists each node's occurrence.
void IndexNodes(CookPlan &);
// The node's index among the nodes of `parent`'s subgraph, or of the planned graph for NoNode, or NoNode.
// An entity a page shows twice, a subgraph's boundaries, gives its first node.
uint32_t IndexOf(const CookPlan &, uint32_t parent, state::Entity);
inline uint32_t IndexOf(const CookPlan &plan, state::Entity node) { return IndexOf(plan, NoNode, node); }
// The index of the node occurrence, or NoNode.
uint32_t IndexOf(const CookPlan &, std::span<const state::Entity> path);
inline std::span<const state::Entity> PathOf(const CookPlan &plan, uint32_t node) { return std::span{plan.Paths}.subspan(plan.PathFirst[node], plan.PathFirst[node + 1] - plan.PathFirst[node]); }

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
inline std::vector<bool> Upstream(const CookPlan &plan, uint32_t node, auto &&follows) { return Upstream(plan.InputFirst, std::span{plan.Inputs}, node, follows); }
// The marked nodes and each node that reads one through any input.
inline std::vector<bool> Downstream(const CookPlan &plan, std::vector<bool> marked) {
    for (const auto n : plan.Order) {
        for (const auto &link : NodeLinks(plan, n)) marked[n] = marked[n] || (link.FromNode != NoNode && marked[link.FromNode]);
    }
    return marked;
}

// The first geometry socket in the direction, or none.
inline std::optional<uint16_t> MainSocket(uint16_t kind, SocketDirection direction) {
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
        if (sockets[s].Types == TypeBit(DataType::Geometry) && sockets[s].Direction == direction) return s;
    }
    return {};
}

struct GraphInstances {
    std::vector<state::Entity> Entities;
};

// The mesh entity has no object, transform or name, and is destroyed when the graph stops instancing its geometry or is removed.
struct GraphOwnedMesh {
    state::Entity MeshEntity;
};

struct PublishSelection {}; // Persistent on a node a viewport operator added, which selects its output as the operator does.

state::Entity CreateNode(state::Registry &, uint16_t kind);
// Sets the mesh entity's graph to the root node of the kind alone, displayed, active and selected.
void RootGraph(state::Registry &, state::Entity mesh_entity, state::Entity root, uint16_t kind);

// Faulted nodes pass through their input, as do unchanged nodes and nodes without geometry.
enum class NodeFault : uint8_t {
    Selection,
    Parameter,
    Attribute, // An attribute input is missing or has the wrong type or domain
    NotClosed,
    NotCoplanar, // A region input's face edges leave the first region's plane
    Empty,
    NoMesh, // An Object Info node refers to no mesh it can read, or an Instance input is unlinked
    Unresolved, // No declaration of an op node takes its input types
    NoValue, // An op node's output is a string, filename, shader or material
    Unavailable, // An op node reads what its consumer's domain lacks
    Coordinates, // A filter's input field reads more than one set of texture coordinates
    Ambiguous, // A native source reads what several domains store and its consumer's domain lacks
    Unstored, // An op output has no attribute type to store it on its consumer's domain
    OutputLimit, // A consumer reads more op outputs than one evaluation writes
    Mismatch, // An input of a node other than an op reads an op output of a type it does not take, or one that varies where it takes a single value
    NotFound, // A geompropvalue op finds no attribute of its name and type, and reads its default
};

// Persistent on a Source node: the mesh storage state, identified by its hash, in which record StoreId holds the node's geometry.
// A pending state is zero, and its record holds the geometry until the node's history step ends.
// A state without a record is empty geometry.
struct SourceState {
    store::Hash128 State{};
    uint32_t StoreId{InvalidOffset};
    bool operator==(const SourceState &) const = default;
};
// What the chained display path's cook keeps for each of its nodes.
// Input is the path node whose result this node ran on, or empty for the root.
// The path's last node holds a pending zero state, and the record holds that node's result as it stands.
struct NodeResult {
    NodePath Input{};
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
// Graph is the mesh entity whose graph holds the edited node's occurrence, and Mesh the mesh entity of the edited input.
// Port is the selection port of the subgraph node Node whose own selection is edited, or none for Node's inline selection.
struct SelectionEditing {
    state::Entity Node, Graph, Mesh, Placement;
    InteractionMode PriorMode;
    Element PriorElement;
    std::optional<InlineSelection> PriorSelection;
    std::optional<uint16_t> Port;
};
// Path holds the subgraph nodes enclosing the node's occurrence in the graph of Object's mesh.
struct OperationHandle {
    state::Entity Node{state::Null}, Object{state::Null};
    HandleParameter Parameter{};
    NodePath Path{};
};
// The transform from each Mesh 2's space into Mesh 1's that a Boolean node's result was cooked with.
// Operand is the object whose transform placed that Mesh 2, or null for a Mesh 2 in this graph's mesh space.
// A graph whose placing objects only moved cooks when this transform changed.
struct OperandPlacement {
    state::Entity Operand{state::Null};
    mat4 ToFirst{1.f};
    bool operator==(const OperandPlacement &) const = default;
};
struct BooleanPlacement {
    std::vector<OperandPlacement> Operands; // One per Mesh 2 link, in link order
    bool operator==(const BooleanPlacement &) const = default;
};
// What a mesh's cooks kept for a node occurrence they evaluated.
// Results and extrusions hold for the nodes of the chained display path, an error for a node that faulted, a handle for a node with a viewport handle, and a placement for each Boolean node a cook evaluated.
// Persistent on the node of an occurrence in the mesh's own graph.
struct NodeError {
    NodeFault Fault;
    bool operator==(const NodeError &) const = default;
};
// Orders occurrence paths, and finds one by any span of entities.
struct PathOrder {
    using is_transparent = void;
    bool operator()(std::span<const state::Entity> a, std::span<const state::Entity> b) const { return std::ranges::lexicographical_compare(a, b); }
};
// Persistent on a subgraph node of a mesh's own graph: what the mesh's cooks kept for the occurrences inside it, by the path after the subgraph node.
struct NodeErrors {
    std::map<NodePath, NodeError, PathOrder> Values;
};
struct NodeResults {
    std::map<NodePath, NodeResult, PathOrder> Values;
};
struct NodeExtrusions {
    std::map<NodePath, NodeExtrusion, PathOrder> Values;
};
struct NodeHandles {
    std::map<NodePath, NodeHandle, PathOrder> Values;
};
struct BooleanPlacements {
    std::map<NodePath, BooleanPlacement, PathOrder> Values;
};
template<typename T> struct InnerOf;
template<> struct InnerOf<NodeError> {
    using Type = NodeErrors;
};
template<> struct InnerOf<NodeResult> {
    using Type = NodeResults;
};
template<> struct InnerOf<NodeExtrusion> {
    using Type = NodeExtrusions;
};
template<> struct InnerOf<NodeHandle> {
    using Type = NodeHandles;
};
template<> struct InnerOf<BooleanPlacement> {
    using Type = BooleanPlacements;
};
template<typename T> using Inner = typename InnerOf<T>::Type;
using OccurrenceStates = TypeList<NodeError, NodeResult, NodeExtrusion, NodeHandle, BooleanPlacement>;
// The value the mesh's cooks kept for the occurrence, or null.
template<typename T> const T *CookedOf(const auto &r, std::span<const state::Entity> path) {
    if (path.size() == 1u) return r.template try_get<const T>(path.front());
    const auto *held = r.template try_get<const Inner<T>>(path.front());
    if (!held) return nullptr;
    const auto found = held->Values.find(path.subspan(1u));
    return found != held->Values.end() ? &found->second : nullptr;
}
// Keeps the value for the occurrence, writing it only when it differs.
// The cook writes what it keeps outside an action's Apply, as engine write-back.
template<typename T> void Keep(auto &r, std::span<const state::Entity> path, T value) {
    if (const auto *held = CookedOf<T>(r, path); held && *held == value) return;
    if (path.size() == 1u) return void(r.template emplace_or_replace<T>(path.front(), std::move(value)));
    if (!r.template all_of<Inner<T>>(path.front())) r.template emplace<Inner<T>>(path.front());
    r.template patch<Inner<T>>(path.front(), [&](auto &kept) { kept.Values.insert_or_assign(NodePath{path.begin() + 1, path.end()}, std::move(value)); });
}
template<typename... T> void Drop(auto &r, std::span<const state::Entity> path) {
    const auto drop = [&]<typename D> {
        if (!CookedOf<D>(r, path)) return;
        if (path.size() == 1u) r.template remove<D>(path.front());
        else if (r.template get<const Inner<D>>(path.front()).Values.size() == 1u) r.template remove<Inner<D>>(path.front());
        else r.template patch<Inner<D>>(path.front(), [&](auto &kept) { kept.Values.erase(kept.Values.find(path.subspan(1u))); });
    };
    (drop.template operator()<T>(), ...);
}
struct StagedNode {}; // Derived on a node a preview step appended, which the stage's later steps edit.

// What a word range of an op value block holds: a literal an op input of the block's graph is set to, a declaration's default, a constant an op input's link holds, a literal of an op of a subgraph, which its subgraph's block holds, or the default of a read of a field from a node other than an op.
// A constant is a subgraph port's value or default an op input reads.
enum class OpSlotKind : uint8_t { Literal,
                                  Default,
                                  Constant,
                                  Copy,
                                  Read };
struct OpSlot {
    NodePath Node;
    std::string Input;
    mx::Type Type;
    OpSlotKind Kind;
    uint32_t Offset; // In words from the block's start
    bool operator==(const OpSlot &) const = default;
};
// Persistent on a geometry graph's mesh entity with op nodes: its block of the op value store, which holds the values of each op occurrence of its plan in plan order.
// Persistent on a subgraph entity with op literals: its block, which holds them by node.
struct OpValueBlock {
    Range Block;
    std::vector<OpSlot> Slots; // By offset
};
// The value store of the registry of built-in subgraphs: the words of each literal of a built-in subgraph's op, by node and input.
// A document's blocks copy them for each occurrence.
struct BuiltinSubgraphLiterals {
    std::map<std::pair<state::Entity, std::string>, std::vector<uint32_t>> Words;
};
} // namespace graph

template<> inline constexpr field::SpecTable<graph::Boolean> Specs<graph::Boolean>{
    {&graph::Boolean::Operation, {.Help = "Set operation applied to Mesh 1 and each Mesh 2 in turn. Difference subtracts every Mesh 2 from Mesh 1."}},
    {&graph::Boolean::Space, {.Help = "Object whose world transform places Mesh 1 and whose local space the result is in, or none to leave Mesh 1 in place. Each Mesh 2 is placed by the object its geometry came from."}},
};
template<> inline constexpr field::SpecTable<graph::Minkowski> Specs<graph::Minkowski>{{&graph::Minkowski::Operation, {.Help = "Sum sweeps Mesh 1 by Mesh 2, and Difference erodes Mesh 1 by it. Two regions sweep in Mesh 1's plane, with Mesh 2's plane through the origin."}}};
template<> inline constexpr field::SpecTable<graph::LevelSet> Specs<graph::LevelSet>{
    {&graph::LevelSet::Min, {.Speed = 0.01f, .Help = "Lower corner of the box the surface is marched over, in local space"}},
    {&graph::LevelSet::Max, {.Speed = 0.01f, .Help = "Upper corner of the box, in local space"}},
    {&graph::LevelSet::EdgeLength, {.Min = 1e-4f, .Speed = 0.001f, .Help = "Largest spacing of the lattice the Value field is read on, which bounds the triangles' edge lengths"}},
    {&graph::LevelSet::Level, {.Speed = 0.01f, .Help = "Value the surface is extracted at, with larger values inside"}},
};
template<> inline constexpr field::SpecTable<graph::StoreNamedAttribute> Specs<graph::StoreNamedAttribute>{
    {&graph::StoreNamedAttribute::Name, {.Help = "Name of the attribute to write. An existing attribute with this name is replaced."}},
    {&graph::StoreNamedAttribute::Domain, {.Help = "Element domain the value is evaluated on and stored on. A field of another domain converts to it."}},
};
template<> inline constexpr field::SpecTable<graph::StoreSelection> Specs<graph::StoreSelection>{{&graph::StoreSelection::Name, {.Help = "Name of the Boolean attribute to write. An existing attribute with this name is replaced."}}};
template<> inline constexpr field::SpecTable<graph::Bake> Specs<graph::Bake>{
    {&graph::Bake::UvSet, {.Min = 0, .Max = 3, .Label = "UV set", .Help = "Texture coordinate set whose layout places the texels on the geometry"}},
    {&graph::Bake::Width, {.Min = 1, .Max = 16384, .Help = "Texels along U"}},
    {&graph::Bake::Height, {.Min = 1, .Max = 16384, .Help = "Texels along V"}},
    {&graph::Bake::Margin, {.Min = 0, .Max = 4096, .SoftMin = 0, .SoftMax = 64, .Help = "Distance in texels past the layout's edges over which texels extend the field from the nearest point of the nearest triangle, so filtered lookups near seams read no empty texels"}},
};
template<> inline constexpr field::SpecTable<graph::SetPosition> Specs<graph::SetPosition>{{&graph::SetPosition::Offset, {.Help = "Offset each selected vertex by the Position input instead of replacing its position"}}};
template<> inline constexpr field::SpecTable<graph::TransformElements> Specs<graph::TransformElements>{
    {&graph::TransformElements::Delta, {.Help = "Translation, rotation, and scale applied to the selected vertices, all by default, around Pivot"}},
    {&graph::TransformElements::Pivot, {.Help = "Center of rotation and scale in Frame space, in local units"}},
    {&graph::TransformElements::Frame, {.Help = "Transform from mesh-local positions into the space Delta applies in"}},
    {&graph::TransformElements::Mirror, {.Speed = 0.01f, .Help = "Normal of the plane through Pivot the selected vertices reflect across before Delta, in Frame space. A zero normal reflects nothing."}},
};
template<> inline constexpr field::SpecTable<graph::ToSphere> Specs<graph::ToSphere>{
    {&graph::ToSphere::Factor, {.Min = 0, .Max = 1, .Speed = 0.01f}},
    {&graph::ToSphere::Center, {.Speed = 0.01f, .Help = "Sphere center in Frame space"}},
    {&graph::ToSphere::Radius, {.Min = 0, .Speed = 0.01f, .Help = "Distance from Center the vertices blend toward, in local units. Zero takes the selected vertices' mean distance."}},
    {&graph::ToSphere::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::PushPull> Specs<graph::PushPull>{
    {&graph::PushPull::Center, {.Speed = 0.01f, .Help = "Point the vertices move toward, in Frame space"}},
    {&graph::PushPull::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::Shear> Specs<graph::Shear>{
    {&graph::Shear::Angle, {.Min = -2 * std::numbers::pi_v<float>, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians}},
    {&graph::Shear::Center, {.Speed = 0.01f, .Help = "Point on the shear plane, in Frame space"}},
    {&graph::Shear::Direction, {.Speed = 0.01f, .Help = "Displacement direction in Frame space"}},
    {&graph::Shear::Normal, {.Speed = 0.01f, .Help = "Shear plane normal in Frame space"}},
    {&graph::Shear::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::Warp> Specs<graph::Warp>{
    {&graph::Warp::Angle, {.Unit = FieldUnit::Radians}},
    {&graph::Warp::OffsetAngle, {.Unit = FieldUnit::Radians}},
    {&graph::Warp::Center, {.Speed = 0.01f, .Help = "Warp center in Frame space"}},
    {&graph::Warp::Orientation, {.Help = "Rotation of the warp plane in Frame space"}},
    {&graph::Warp::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::Bend> Specs<graph::Bend>{
    {&graph::Bend::Angle, {.Unit = FieldUnit::Radians}},
    {&graph::Bend::OffsetAngle, {.Unit = FieldUnit::Radians}},
    {&graph::Bend::Center, {.Speed = 0.01f, .Help = "Bend start in Frame space"}},
    {&graph::Bend::Orientation, {.Help = "Rotation of the bend plane in Frame space"}},
    {&graph::Bend::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::VertexSlide> Specs<graph::VertexSlide>{
    {&graph::VertexSlide::Direction, {.Speed = 0.01f, .Help = "Direction in Frame space that chooses each vertex's edge"}},
    {&graph::VertexSlide::Reference, {.Help = "Vertex the even slide measures from, or the first selected vertex"}},
    {&graph::VertexSlide::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::EdgeSlide> Specs<graph::EdgeSlide>{
    {&graph::EdgeSlide::Direction, {.Speed = 0.01f, .Help = "Direction in Frame space that chooses the positive side"}},
    {&graph::EdgeSlide::Reference, {.Help = "Handle of the vertex where Direction chooses the positive side, or the maximum for the first selected vertex"}},
    {&graph::EdgeSlide::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::Flatten> Specs<graph::Flatten>{
    {&graph::Flatten::Factor, {.Min = 0, .Max = 1}},
    {&graph::Flatten::ViewNormal, {.Speed = 0.01f, .Help = "Flattening direction of the View method, in Frame space"}},
    {&graph::Flatten::Frame, {.Help = "Transform from mesh-local positions into the space the operation applies in"}},
};
template<> inline constexpr field::SpecTable<graph::MarkSharp> Specs<graph::MarkSharp>{{&graph::MarkSharp::Sharp, {.Help = "Mark the elements sharp, or clear their marks"}}};
template<> inline constexpr field::SpecTable<graph::Knife> Specs<graph::Knife>{
    {&graph::Knife::Start, {.Speed = 1.f, .Help = "First end of the cut segment, in view pixels from the top left"}},
    {&graph::Knife::End, {.Speed = 1.f, .Help = "Second end of the cut segment, in view pixels from the top left"}},
    {&graph::Knife::Extent, {.Min = 1, .Speed = 1.f, .Help = "Width and height of the view in pixels"}},
    {&graph::Knife::ViewProjection, {.Help = "Transform from Frame space into the view's clip space"}},
    {&graph::Knife::Frame, {.Help = "Transform from mesh-local positions into the space ViewProjection projects"}},
};
template<> inline constexpr field::SpecTable<graph::Shade> Specs<graph::Shade>{
    {&graph::Shade::Mode, {.Help = "Shade every face flat, smooth with no sharp marks, or smooth with sharp marks where faces meet above Angle"}},
    {&graph::Shade::Angle, {.Min = 0, .Max = std::numbers::pi, .Unit = FieldUnit::Radians, .Help = "Largest angle between adjacent face normals that shades smooth"}},
};
template<> inline constexpr field::SpecTable<graph::InstanceOnPoints> Specs<graph::InstanceOnPoints>{
    {&graph::InstanceOnPoints::Prototype, {.Help = "Scene subgraph that an instance at each point shows. When empty, use the Instance geometry input."}},
    {&graph::InstanceOnPoints::Scale, {.Help = "Scale of each instance while the Scale input is unlinked"}},
    {&graph::InstanceOnPoints::Rotation, {.Help = "Rotation of each instance while the Rotation input is unlinked"}},
};
template<> inline constexpr field::SpecTable<graph::ObjectInfo> Specs<graph::ObjectInfo>{{&graph::ObjectInfo::Object, {.Help = "Object whose mesh the node emits. The object this graph edits is not allowed."}}};

template<> inline constexpr field::SpecTable<graph::Extrude> Specs<graph::Extrude>{
    {&graph::Extrude::Mode, field::SpecOf<&action::mesh::Extrude::Mode>},
    {&graph::Extrude::Offset, field::SpecOf<&action::mesh::Extrude::Offset>},
    {&graph::Extrude::OffsetScale, field::SpecOf<&action::mesh::Extrude::OffsetScale>},
    {&graph::Extrude::AlongNormals, field::SpecOf<&action::mesh::Extrude::AlongNormals>},
    {&graph::Extrude::Steps, field::SpecOf<&action::mesh::Revolve::Steps>},
    {&graph::Extrude::Angle, field::SpecOf<&action::mesh::Revolve::Angle>},
    {&graph::Extrude::Axis, field::SpecOf<&action::mesh::Revolve::Axis>},
    {&graph::Extrude::Center, field::SpecOf<&action::mesh::Revolve::Center>},
};
template<> inline constexpr field::SpecTable<graph::SubdivisionSurface> Specs<graph::SubdivisionSurface>{{&graph::SubdivisionSurface::Levels, {.Min = 1, .Max = 6, .Help = "Number of subdivisions, each quadrupling the face count"}}};
template<> inline constexpr field::SpecTable<graph::SmoothSurface> Specs<graph::SmoothSurface>{{&graph::SmoothSurface::Cuts, {.Min = 1, .Max = 16, .Help = "Number of new vertices along each edge"}}};
template<> inline constexpr field::SpecTable<graph::Project> Specs<graph::Project>{
    {&graph::Project::Point, {.Speed = 0.01f, .Help = "Point on the plane the outline lies on, in local space"}},
    {&graph::Project::Normal, {.Speed = 0.01f, .Help = "Direction the mesh is seen along, and the plane's normal, in local space"}},
};
template<> inline constexpr field::SpecTable<graph::Bisect> Specs<graph::Bisect>{
    {&graph::Bisect::Point, field::SpecOf<&action::mesh::Bisect::Point>},
    {&graph::Bisect::Normal, field::SpecOf<&action::mesh::Bisect::Normal>},
    {&graph::Bisect::ClearInner, field::SpecOf<&action::mesh::Bisect::ClearInner>},
    {&graph::Bisect::ClearOuter, field::SpecOf<&action::mesh::Bisect::ClearOuter>},
};
