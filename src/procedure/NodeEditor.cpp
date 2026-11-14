#include "procedure/NodeEditor.h"
#include "procedure/Macros.h"
#include "scene/HierarchyInstance.h"

#include "action/Action.h"
#include "action/View.h"
#include "mesh/Mesh.h"
#include "procedure/NodeTheme.h"
#include "procedure/NodeView.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceUi.h"
#include "viewport/FrameState.h"

#include <imgui_internal.h>

#include <bit>
#include <numeric>

namespace procedure {
using namespace ImGui;
using numeric::uvec2;
using numeric::vec2;

namespace {
constexpr float WireStraightening{3.5f};
constexpr float MinZoom{.25f}, MaxZoom{3.f};
constexpr float TrackpadPoints{10.f}; // The platform reports a trackpad's scroll at a tenth of its points.
constexpr float ZoomStep{1.1f}, TrackpadStepPoints{100.f}; // A zoom step is one mouse wheel notch, or 100 points of trackpad scroll.

enum class KindGroup : uint8_t { Sources,
                                 Operators,
                                 Combine,
                                 Attributes,
                                 Math,
                                 Values,
                                 Generators,
                                 Instances };
constexpr const char *GroupNames[]{"Sources", "Operators", "Combine", "Attributes", "Math", "Values", "Generators", "Instances"};
using C = NodeThemeColors;
constexpr vec4 C::*GroupColors[]{&C::SourcesTitle, &C::OperatorsTitle, &C::CombineTitle, &C::AttributesTitle, &C::MathTitle, &C::ValuesTitle, &C::GeneratorsTitle, &C::InstancesTitle};
using AttributeKinds = TypeList<Position, Normal, Curvature, Boundary, PlaneSide, Linked, Adjacent, EdgeLoop, EdgeRing, Index, NamedAttribute, StoreNamedAttribute, StoreSelection, SelectionAttribute, SetPosition>;
using MathKinds = TypeList<FloatMath, VectorMath, Compare, BooleanMath>;
using InstanceKinds = TypeList<MeshToPoints, InstanceOnPoints, RealizeInstances>;
template<typename K> constexpr KindGroup GroupOf() {
    if constexpr (std::same_as<K, Source> || std::same_as<K, ObjectInfo>) return KindGroup::Sources;
    else if constexpr (std::same_as<K, Join> || std::same_as<K, Boolean> || std::same_as<K, Minkowski>) return KindGroup::Combine;
    else if constexpr (Generator<K> || std::same_as<K, LevelSet>) return KindGroup::Generators;
    else if constexpr (InTypeList<K, AttributeKinds>) return KindGroup::Attributes;
    else if constexpr (InTypeList<K, MathKinds>) return KindGroup::Math;
    else if constexpr (ValueKind<K>) return KindGroup::Values;
    else if constexpr (InTypeList<K, InstanceKinds>) return KindGroup::Instances;
    else return KindGroup::Operators;
}
constexpr auto KindGroups{[]<typename... Ks>(TypeList<Ks...>) { return std::array{GroupOf<Ks>()...}; }(NodeKinds{})};
constexpr auto KindTitles{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Title<Ks>.c_str()...}; }(NodeKinds{})};

template<typename K> constexpr const char *Help{};
template<> constexpr const char *Help<Source>{"Emits the mesh's source geometry"};
template<> constexpr const char *Help<Join>{"Joins every linked geometry into one mesh, in link order"};
template<> constexpr const char *Help<LevelSet>{"Marches the surface where a field crosses a level over a box"};
template<> constexpr const char *Help<Boolean>{"Combines Mesh 1 with each Mesh 2 in turn by union, difference, or intersection. The inputs are closed meshes or regions on one plane."};
template<> constexpr const char *Help<Minkowski>{"Sweeps one closed mesh or region by another, dilating or eroding it"};
template<> constexpr const char *Help<primitive::Plane>{"Generates a grid of quads in the XZ plane, facing +Y"};
template<> constexpr const char *Help<primitive::Circle>{"Generates a circle in the XZ plane, facing +Y"};
template<> constexpr const char *Help<primitive::Cube>{"Generates a box centered on the origin"};
template<> constexpr const char *Help<primitive::IcoSphere>{"Generates a sphere by subdividing an icosahedron"};
template<> constexpr const char *Help<primitive::UVSphere>{"Generates a sphere of slices around the Y axis and stacks along it"};
template<> constexpr const char *Help<primitive::QuadSphere>{"Generates a sphere of quads by mapping a subdivided cube onto it"};
template<> constexpr const char *Help<primitive::Torus>{"Generates a torus around the Y axis"};
template<> constexpr const char *Help<primitive::Cylinder>{"Generates a cylinder, frustum or cone along the Y axis, centered on the origin"};
template<> constexpr const char *Help<action::mesh::Delete>{"Deletes the selected vertices, edges, or faces"};
template<> constexpr const char *Help<action::mesh::Merge>{"Merges the selected vertices into one, or merges nearby pairs"};
template<> constexpr const char *Help<Extrude>{"Extrudes the selected elements in steps, rotating about an axis and moving by the offset, joined to the geometry they came from"};
template<> constexpr const char *Help<action::mesh::Duplicate>{"Duplicates the selected faces in place onto copied vertices"};
template<> constexpr const char *Help<action::mesh::Split>{"Disconnects the selected faces from the unselected geometry around them"};
template<> constexpr const char *Help<action::mesh::Dissolve>{"Removes the selected elements and joins the faces around them"};
template<> constexpr const char *Help<action::mesh::Subdivide>{"Cuts the selected edges and splits their faces into smaller faces"};
template<> constexpr const char *Help<action::mesh::SplitFaces>{"Splits the selected faces until each part is a triangle, is convex, or is planar within Angle"};
template<> constexpr const char *Help<action::mesh::TrisToQuads>{"Joins pairs of selected triangles into quads"};
template<> constexpr const char *Help<action::mesh::Poke>{"Fans each selected face around a new center vertex"};
template<> constexpr const char *Help<action::mesh::FlipNormals>{"Reverses the winding and normals of the selected faces"};
template<> constexpr const char *Help<action::mesh::EdgeSplit>{"Splits the mesh along the selected edges, disconnecting the faces on either side"};
template<> constexpr const char *Help<action::mesh::Inset>{"Insets the selected faces, as one region or each face on its own"};
template<> constexpr const char *Help<action::mesh::Solidify>{"Thickens the selected faces into a shell, with a rim along the region boundary"};
template<> constexpr const char *Help<action::mesh::Wireframe>{"Builds solid struts along the edges of the selected faces"};
template<> constexpr const char *Help<action::mesh::Bevel>{"Bevels the selected edges or vertices"};
template<> constexpr const char *Help<action::mesh::SmoothVertices>{"Moves the selected vertices toward the mean of their edge neighbors"};
template<> constexpr const char *Help<action::mesh::ShrinkFatten>{"Moves the selected vertices along their normals"};
template<> constexpr const char *Help<action::mesh::Randomize>{"Offsets the selected vertices by repeatable random amounts"};
template<> constexpr const char *Help<action::mesh::SpaceEvenly>{"Spaces the vertices of the selected edge chains evenly along them"};
template<> constexpr const char *Help<action::mesh::RelaxEdgeLoops>{"Smooths the selected edge chains, keeping their open endpoints and junctions"};
template<> constexpr const char *Help<action::mesh::CurveBetweenSelected>{"Moves the unselected loop vertices onto a curve through the selected ones"};
template<> constexpr const char *Help<action::mesh::Circularize>{"Fits the selected boundaries and wire chains to circles"};
template<> constexpr const char *Help<action::mesh::MakePlanarFaces>{"Flattens the selected faces toward their planes"};
template<> constexpr const char *Help<Flatten>{"Flattens the selected face regions and edge chains onto planes"};
template<> constexpr const char *Help<ToSphere>{"Blends the selected vertices toward a sphere around Center"};
template<> constexpr const char *Help<PushPull>{"Moves the selected vertices toward or away from Center"};
template<> constexpr const char *Help<Shear>{"Shears the selected vertices along Direction"};
template<> constexpr const char *Help<Warp>{"Warps the selected vertices around Center"};
template<> constexpr const char *Help<Bend>{"Bends the selected vertices from Center"};
template<> constexpr const char *Help<VertexSlide>{"Slides each selected vertex along its incident edge nearest Direction"};
template<> constexpr const char *Help<EdgeSlide>{"Slides the selected edge chains along their adjacent faces"};
template<> constexpr const char *Help<action::mesh::SnapSymmetry>{"Snaps the selected vertices to their mirrored partners across an axis plane"};
template<> constexpr const char *Help<Project>{"Leaves the outline of the mesh seen along a direction, as a region on a plane"};
template<> constexpr const char *Help<Bisect>{"Cuts the mesh along a plane and deletes the vertices beyond it on the cleared sides"};
template<> constexpr const char *Help<action::mesh::ConvexHull>{"Replaces the selection with the convex hull of the selected vertices"};
template<> constexpr const char *Help<action::mesh::Simplify>{"Collapses edges within a tolerance, then toward a triangle ratio"};
template<> constexpr const char *Help<action::mesh::RecalculateNormals>{"Makes the selected faces' normals point outside each closed surface they belong to"};
template<> constexpr const char *Help<action::mesh::ConnectVertices>{"Splits faces along chords between their consecutive selected vertices"};
template<> constexpr const char *Help<action::mesh::Fill>{"Creates an edge from two selected vertices, fills the selected loops of at most Sides edges, or forms a face from the selected points"};
template<> constexpr const char *Help<action::mesh::BridgeEdgeLoops>{"Bridges two selected edge loops or chains with a strip of faces"};
template<> constexpr const char *Help<action::mesh::GridFill>{"Fills one selected closed loop with a grid of quads"};
template<> constexpr const char *Help<action::mesh::EdgeRotate>{"Rotates each selected edge between its two faces"};
template<> constexpr const char *Help<action::mesh::BeautifyFaces>{"Rearranges the selected triangles into more evenly shaped ones"};
template<> constexpr const char *Help<SmoothSurface>{"Refines a closed mesh onto smooth patches through its shading normals, so flat-shaded faces and sharp edges stay creased"};
template<> constexpr const char *Help<SubdivisionSurface>{"Smooths the whole mesh by Catmull-Clark subdivision"};
template<> constexpr const char *Help<action::mesh::Unsubdivide>{"Reverses subdivision by removing alternate selected vertices of regular regions"};
template<> constexpr const char *Help<Knife>{"Cuts every edge whose projection in a view crosses a screen segment"};
template<> constexpr const char *Help<action::mesh::CycleCornerAttribute>{"Shifts or reverses the UVs or colors around the corners of the selected faces"};
template<> constexpr const char *Help<MarkSharp>{"Marks the selected edges sharp or smooth, or shades the selected faces flat or smooth"};
template<> constexpr const char *Help<Shade>{"Shades every face flat, smooth, or smooth with sharp edges where adjacent faces meet at more than Angle"};
template<> constexpr const char *Help<TransformElements>{"Moves, rotates, and scales the selected vertices, all by default, around a pivot, after reflecting them across a mirror plane through it"};
template<> constexpr const char *Help<Position>{"Emits each vertex's position"};
template<> constexpr const char *Help<Normal>{"Emits each vertex's or face's normal"};
template<> constexpr const char *Help<Curvature>{"Emits each vertex's Gaussian or mean curvature"};
template<> constexpr const char *Help<Boundary>{"Selects the mesh boundary in a domain"};
template<> constexpr const char *Help<PlaneSide>{"Selects the elements in front of a plane, behind it, or on it"};
template<> constexpr const char *Help<Linked>{"Selects the connected components holding the selection"};
template<> constexpr const char *Help<Adjacent>{"Grows or shrinks the selection by rings of adjacent elements"};
template<> constexpr const char *Help<EdgeLoop>{"Selects the edge loops through the selected edges"};
template<> constexpr const char *Help<EdgeRing>{"Selects the edge rings through the selected edges"};
template<> constexpr const char *Help<SelectionAttribute>{"Emits the selection as a Boolean attribute in its domain"};
template<> constexpr const char *Help<Index>{"Emits each element's index in its domain, counting live elements only"};
template<> constexpr const char *Help<NamedAttribute>{"Emits the values of the attribute with the given name and type"};
template<> constexpr const char *Help<VectorMath>{"Applies a vector operation to A and B per element"};
template<> constexpr const char *Help<FloatMath>{"Applies a math operation to its operands, per element"};
template<> constexpr const char *Help<Compare>{"Emits the result of comparing A to B per element"};
template<> constexpr const char *Help<BooleanMath>{"Applies a logical operation to A and B per element"};
template<> constexpr const char *Help<StoreNamedAttribute>{"Writes the Value input to the named attribute of the geometry"};
template<> constexpr const char *Help<StoreSelection>{"Writes the selection to the named Boolean attribute of the geometry"};
template<> constexpr const char *Help<SetPosition>{"Sets or offsets the selected vertices' positions from a vector attribute"};
template<> constexpr const char *Help<MeshToPoints>{"Emits the input's vertices as points"};
template<> constexpr const char *Help<InstanceOnPoints>{"Places an instance of the second input at each point of the first"};
template<> constexpr const char *Help<RealizeInstances>{"Emits a copy of the instance geometry for each instance, placed by its transform"};
template<> constexpr const char *Help<ObjectInfo>{"Emits another object's mesh in that mesh's local space"};
template<> constexpr const char *Help<FloatValue>{"Emits a constant float"};
template<> constexpr const char *Help<IntegerValue>{"Emits a constant unsigned integer"};
template<> constexpr const char *Help<BooleanValue>{"Emits a constant boolean"};
template<> constexpr const char *Help<VectorValue>{"Emits a constant vector"};
template<> constexpr const char *Help<RotationValue>{"Emits a constant rotation"};
template<> constexpr const char *Help<Subnet>{"Contains a network with named inputs and outputs"};
constexpr auto KindHelp{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Help<Ks>...}; }(NodeKinds{})};
static_assert(std::ranges::none_of(KindHelp, [](const char *help) { return help == nullptr; }), "Every node kind needs a Help sentence.");

constexpr const char *FaultTexts[]{"The selection does not fit this operation.", "A parameter is invalid.", "An attribute input is missing or has the wrong type or domain.", "An input mesh is not a closed surface.", "The boolean produced no geometry.", "The node reads no mesh: the Object Info object has none, or the Instance input is unlinked."};

constexpr const char *SocketLabels[]{"Geometry", "Selection", "Attribute", "Instances", "Value", "Reference"};
const char *LabelOf(Socket socket) { return socket.Name ? socket.Name : SocketLabels[uint8_t(socket.Type)]; }
constexpr vec4 C::*TypeColors[]{&C::FloatSocket, &C::MixedSocket, &C::VectorSocket, &C::RotationSocket, &C::IntegerSocket, &C::BooleanSocket};
vec4 SocketColor(const C &colors, Socket socket) {
    if (socket.Type == SocketType::Geometry) return colors.GeometrySocket;
    if (socket.Type == SocketType::Instances) return colors.InstancesSocket;
    return std::has_single_bit(socket.Types) ? colors.*TypeColors[std::countr_zero(socket.Types)] : colors.MixedSocket;
}

constexpr const char *AttributeTypeNames[]{"Float", "2D Vector", "Vector", "Rotation", "Integer", "Boolean"};
static_assert(std::size(AttributeTypeNames) == AttributeTypeCount);
std::array<std::string, 512> SocketTypeTexts;
// A socket's direction and attribute types, such as "Input, Float or Vector", built once per direction and types mask.
const char *SocketTypeText(Socket socket) {
    const bool input = socket.Direction == SocketDirection::In;
    if (socket.Types == 0u) {
        if (socket.Type == SocketType::Instances) return input ? "Input, Instances" : "Output, Instances";
        return input ? "Input, Geometry" : "Output, Geometry";
    }
    auto &text = SocketTypeTexts[(input ? 0u : 256u) + socket.Types];
    if (!text.empty()) return text.c_str();
    text = input ? "Input, " : "Output, ";
    if (socket.Types == AnyType) {
        text += "any type";
        return text.c_str();
    }
    for (uint32_t types = socket.Types; types != 0u;) {
        text += AttributeTypeNames[std::countr_zero(types)];
        types &= types - 1u;
        if (types != 0u) text += std::has_single_bit(types) ? " or " : ", ";
    }
    return text.c_str();
}

// The help of the parameter field behind each socket of kind K, or null for a fixed socket.
template<typename K> constexpr const char *FieldHelp(Socket socket) {
    const char *help{};
    if constexpr (field::Walkable<K>) {
        K value{};
        field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
            using F = std::remove_cvref_t<decltype(member)>;
            if (socket.Field != I) return;
            help = Spec<K, I>.Help;
            if constexpr (detail::TypesOf<F>() == 0u && field::Walkable<F>) {
                if (socket.Member != NoField && Specs<F>.ByIndex[socket.Member].Help) help = Specs<F>.ByIndex[socket.Member].Help;
            }
        });
    }
    return help;
}
// Each socket's help, in Sockets order.
constexpr auto SocketHelps{[]<typename... Ks>(TypeList<Ks...>) {
    std::array<const char *, Sockets.size()> helps{};
    (std::ranges::transform(detail::SocketsOf<Ks>(), helps.begin() + SocketFirst[KindOf<Ks>], FieldHelp<Ks>), ...);
    return helps;
}(NodeKinds{})};

ImVec2 Im(vec2 v) { return {v.x, v.y}; }
using ui::U32;
void DrawSocket(ImDrawList &draw, const NodeTheme &theme, ImVec2 at, float radius, float multi_half, Socket socket, vec4 color, bool hovered) {
    const auto fill = U32(color), outline = U32(hovered ? theme.Colors.Hovered : theme.Colors.Outline);
    if (socket.Multi) {
        const ImVec2 half{radius * .8f, multi_half};
        draw.AddRectFilled(at - half, at + half, fill, half.x);
        draw.AddRect(at - half, at + half, outline, half.x, 0, theme.Sizes.OutlineWidth);
        return;
    }
    if (socket.Type == SocketType::Attribute || socket.Type == SocketType::Selection) {
        const float r = radius * 1.2f;
        draw.AddQuadFilled({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, fill);
        draw.AddQuad({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, outline, theme.Sizes.OutlineWidth);
        return;
    }
    draw.AddCircleFilled(at, radius, fill);
    draw.AddCircle(at, radius, outline, 0, theme.Sizes.OutlineWidth);
}

// A node shows inputs down its left edge and outputs down its right, one per cell row under a one-cell title.
// Tray sockets share one row, marked by a chevron, while the tray is closed.
// Each tray socket has its own row while the tray is open.
constexpr uint16_t NoRow{UINT16_MAX};
constexpr float MultiHalf{.38f}, MultiSpacing{.22f}; // A multi-input socket's half height and the maximum spacing between its links, in cells.
constexpr float CanvasMargin{.5f}; // The empty space left of and above cell zero, in cells.

// The labels Blender's Math node gives the operation's three operands.
// Null marks an operand the operation does not use.
constexpr std::array<const char *, 3> OperandLabels(FloatMathOp op) {
    using enum FloatMathOp;
    switch (op) {
        case Add:
        case Subtract:
        case Multiply:
        case Divide:
        case Minimum:
        case Maximum:
        case TruncatedModulo:
        case FlooredModulo:
        case Arctan2: return {"Value", "Value", nullptr};
        case MultiplyAdd: return {"Value", "Multiplier", "Addend"};
        case Power: return {"Base", "Exponent", nullptr};
        case Logarithm: return {"Value", "Base", nullptr};
        case LessThan:
        case GreaterThan: return {"Value", "Threshold", nullptr};
        case Compare: return {"Value", "Value", "Epsilon"};
        case SmoothMinimum:
        case SmoothMaximum: return {"Value", "Value", "Distance"};
        case Wrap: return {"Value", "Max", "Min"};
        case Snap: return {"Value", "Increment", nullptr};
        case PingPong: return {"Value", "Scale", nullptr};
        case ToRadians: return {"Degrees", nullptr, nullptr};
        case ToDegrees: return {"Radians", nullptr, nullptr};
        default: return {"Value", nullptr, nullptr};
    }
}
constexpr uint16_t OperandCount{3};
static_assert(ParameterSocket<&FloatMath::B> == 1u && ParameterSocket<&FloatMath::C> == 2u, "A float math node's operands are its first sockets.");
constexpr const char *OperationHeading(FloatMathOp op) {
    using enum FloatMathOp;
    return op == Add ? "Functions" : op == Minimum ? "Comparison" :
        op == Round                                ? "Rounding" :
        op == Sine                                 ? "Trigonometric" :
        op == ToRadians                            ? "Conversion" :
                                                     nullptr;
}

// A grid cell's side in canvas points.
float CellOf(const NodeTheme &theme) { return theme.Sizes.Cell.FollowFrameHeight ? GetFrameHeight() : theme.Sizes.Cell.Points; }
uint32_t CellsOf(float cell, float points) { return std::max(1u, uint32_t(std::ceil(points / cell))); }

const char *OperationTitle(const state::Registry &r, state::Entity node, uint16_t kind) {
    return kind == KindOf<FloatMath> ? field::EnumLabel(r.get<const FloatMath>(node).Operation) : KindTitles[kind];
}
NodeView GeometryNodeDefinitions(const state::Registry &r, const CookPlan &plan, state::Entity network) {
    NodeView view{.Network = network, .Geometry = &plan, .DisplayNode = plan.DisplayNode};
    view.Nodes.reserve(plan.Nodes.size());
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        const auto [entity, kind] = plan.Nodes[n];
        const auto position = entity == network ? NetworkPosition(r, network, {entity}).value_or(uvec2{1u, 1u}) : r.get<const NodeLayout>(entity).Position;
        auto title = kind == KindOf<Subnet> ? GetName(r, entity) : std::string{OperationTitle(r, entity, kind)};
        view.Nodes.push_back({{entity}, std::move(title), KindTitles[kind], kind, position});
    }
    return view;
}
uint32_t ActiveIndex(const state::Registry &r, const NodeView &view) {
    if (view.Geometry) return ActiveIndex(r, *view.Geometry);
    const auto *selection = r.try_get<const NetworkActive>(view.Network);
    const auto active = selection ? selection->Item : NetworkItem{FindActiveEntity(r)};
    const auto at = std::ranges::find(view.Nodes, active, &EditorNode::Item);
    return at == view.Nodes.end() ? NoNode : uint32_t(at - view.Nodes.begin());
}
bool Selected(const state::Registry &r, const NodeView &view, uint32_t node) {
    const auto owner = view.Geometry ? r.get<const ProcedureNode>(view.Nodes[node].Item.Entity).Procedure : view.Network;
    const auto *selection = r.try_get<const NetworkActive>(owner);
    return selection && selection->Contains(view.Nodes[node].Item);
}
std::vector<NetworkItem> ItemsOf(const NodeView &view, std::span<const uint32_t> indices) {
    std::vector<NetworkItem> items;
    items.reserve(indices.size());
    for (const auto n : indices) items.push_back(view.Nodes[n].Item);
    return items;
}

bool IsLinked(const CookPlan &plan, uint32_t node, uint16_t socket) { return plan.Wires[plan.InputFirst[node] + socket].FromNode != NoNode; }
bool IsLinked(const NodeView &view, uint32_t node, uint16_t socket) { return WiresOf(view)[InputsFirst(view)[node] + socket].FromNode != NoNode; }
std::vector<Socket> LinkedParameters(const CookPlan &plan, uint32_t node) {
    const auto [first, end] = ParameterRange(plan, node);
    std::vector<Socket> linked;
    for (auto s = first; s < end; ++s) {
        if (IsLinked(plan, node, s)) linked.push_back(NodeSockets(plan, node)[s]);
    }
    return linked;
}

const FloatMath *MathOf(const state::Registry &r, const NodeView &view, uint32_t node) {
    return view.Geometry && view.Nodes[node].Kind == KindOf<FloatMath> ? &r.get<const FloatMath>(view.Nodes[node].Item.Entity) : nullptr;
}
const char *LabelOf(const state::Registry &r, const NodeView &view, uint32_t node, uint16_t socket) {
    if (const auto *math = MathOf(r, view, node); math && socket < OperandCount) {
        if (const auto *label = OperandLabels(math->Operation)[socket]) return label;
    }
    return LabelOf(NodeSockets(view, node)[socket]);
}
// A float math node hides the operands its operation does not use, unless they are linked.
bool Shows(const state::Registry &r, const NodeView &view, uint32_t node, uint16_t socket) {
    const auto *math = MathOf(r, view, node);
    return !math || socket >= OperandCount || OperandLabels(math->Operation)[socket] || IsLinked(view, node, socket);
}

// The row of each input socket below the node's title, with the tray closed and with it open.
// NoRow marks a hidden input.
// With the tray closed, every tray socket is on row Shown, the row after the other inputs.
struct InputRows {
    std::vector<uint16_t> Closed, Open;
    uint16_t Shown{}, Tray{};
    uint16_t ClosedCount() const { return Shown + uint16_t(Tray > 0u); }
    uint16_t OpenCount() const { return Shown + Tray; }
};
InputRows RowsOf(uint16_t inputs, auto &&in_tray, auto &&shows) {
    InputRows rows{std::vector<uint16_t>(inputs, NoRow), std::vector<uint16_t>(inputs, NoRow)};
    for (uint16_t s = 0u; s < inputs; ++s) {
        if (shows(s) && !in_tray(s)) rows.Closed[s] = rows.Open[s] = rows.Shown++;
    }
    for (uint16_t s = 0u; s < inputs; ++s) {
        if (!shows(s) || !in_tray(s)) continue;
        rows.Closed[s] = rows.Shown;
        rows.Open[s] = rows.Shown + rows.Tray++;
    }
    return rows;
}

// A node's cells: its top left and size, and whether it has inputs and outputs, which keep a free cell beside them for wires.
struct Box {
    uint32_t X, Y;
    uint16_t W, H;
    bool In, Out;
};
// The node's box with its tray closed.
// Its width also fits the labels of the open tray.
// `label(s)` returns the label of socket s.
// A label is at most MaxLabelCells wide and ends in an ellipsis when it is longer.
Box BoxOf(const NodeTheme &theme, float cell, const char *title, std::span<const Socket> sockets, const InputRows &rows, auto &&label) {
    const auto inputs = uint16_t(rows.Closed.size());
    const uint32_t outputs = uint32_t(sockets.size()) - inputs;
    const auto label_cells = [&](uint16_t s) {
        const float indent = s < inputs && rows.Open[s] >= rows.Shown ? theme.Sizes.ParameterIndent : 0.f;
        return std::min(CellsOf(cell, theme.Sizes.SocketRadius + theme.Sizes.LabelSpacing * 2.f + indent + CalcTextSize(label(s)).x), uint32_t(theme.Sizes.MaxLabelCells));
    };
    std::vector<uint32_t> left(std::max<uint32_t>(rows.OpenCount(), outputs));
    for (uint16_t s = 0u; s < inputs; ++s) {
        if (rows.Open[s] != NoRow) left[rows.Open[s]] = label_cells(s);
    }
    uint32_t width = CellsOf(cell, CalcTextSize(title).x + theme.Sizes.TitlePadding * 2.f);
    for (uint32_t row = 0u; row < left.size(); ++row) width = std::max(width, left[row] + (row < outputs ? label_cells(uint16_t(inputs + row)) : 0u));
    return {0, 0, uint16_t(width), uint16_t(1u + std::max({uint32_t(rows.ClosedCount()), outputs, 1u})), rows.ClosedCount() > 0u, outputs > 0u};
}
// Whether the boxes leave a free cell beside every socket column between them: right of a when a has outputs or b inputs, and the reverse.
bool Clear(const Box &a, const Box &b) {
    if (a.Y >= b.Y + b.H || b.Y >= a.Y + a.H) return true;
    return b.X >= a.X + a.W + uint32_t(a.Out || b.In) || a.X >= b.X + b.W + uint32_t(b.Out || a.In);
}
// Whether every box of the group, moved by the offset, stays on the canvas, keeps a free cell left of its inputs, and is clear of every placed box.
bool Fits(std::span<const Box> boxes, const std::vector<bool> &placed, std::span<const uint32_t> group, int64_t dx, int64_t dy) {
    for (const auto m : group) {
        auto box = boxes[m];
        const int64_t x = int64_t(box.X) + dx, y = int64_t(box.Y) + dy;
        if (x < int64_t(box.In) || y < 0) return false;
        box.X = uint32_t(x);
        box.Y = uint32_t(y);
        for (uint32_t n = 0u; n < boxes.size(); ++n) {
            if (placed[n] && !Clear(box, boxes[n])) return false;
        }
    }
    return true;
}
constexpr int64_t PlaceReach{64}; // The number of rings of cells Place searches around the wanted offset.
// The offset nearest the wanted one where the group fits, searching outward one ring of cells at a time.
// If nothing fits within PlaceReach, the group goes to the right of every placed box.
std::pair<int64_t, int64_t> Place(std::span<const Box> boxes, const std::vector<bool> &placed, std::span<const uint32_t> group, int64_t dx, int64_t dy) {
    for (int64_t ring = 0; ring <= PlaceReach; ++ring) {
        for (int64_t i = -ring; i <= ring; ++i) {
            for (int64_t j = -ring; j <= ring; ++j) {
                if ((std::abs(i) == ring || std::abs(j) == ring) && Fits(boxes, placed, group, dx + i, dy + j)) return {dx + i, dy + j};
            }
        }
    }
    int64_t right = 0, left = INT64_MAX, top = INT64_MAX;
    for (uint32_t n = 0u; n < boxes.size(); ++n) {
        if (placed[n]) right = std::max(right, int64_t(boxes[n].X) + boxes[n].W);
    }
    for (const auto m : group) {
        left = std::min(left, int64_t(boxes[m].X));
        top = std::min(top, int64_t(boxes[m].Y));
    }
    return {right + 2 - left, std::max(dy, -top)};
}
// Moves each box, in order, from its node's stored cell to the nearest cell where it fits among the boxes before it.
// Every node position the editor uses comes from here, so nodes never overlap and always keep their wire cells.
void Arrange(std::vector<Box> &boxes) {
    std::vector<bool> placed(boxes.size());
    for (uint32_t n = 0u; n < boxes.size(); ++n) {
        const auto [dx, dy] = Place(boxes, placed, std::array{n}, 0, 0);
        boxes[n].X = uint32_t(int64_t(boxes[n].X) + dx);
        boxes[n].Y = uint32_t(int64_t(boxes[n].Y) + dy);
        placed[n] = true;
    }
}

// The screen layout of the network, with cell c at Origin + c * Step.
struct Graph {
    const NodeView &View;
    const NodeTheme &Theme;
    ImVec2 Origin;
    float Step; // A cell's side in screen points
    float Zoom;
    uint32_t Expanded;
    std::vector<Box> Boxes;
    std::vector<InputRows> Rows;
    uint32_t DetachNode{NoNode}; // The node whose input link in DetachSlot a wire drag has picked up
    uint16_t DetachSlot{};
    uint32_t DropNode{NoNode}; // The node whose multi-input socket DropSocket leaves a gap at DropIndex for the held wire
    uint16_t DropSocket{}, DropIndex{};
};
Graph GraphOf(const state::Registry &r, const NodeView &view, const NodeTheme &theme, ImVec2 origin, float cell, float zoom, uint32_t expanded) {
    Graph graph{view, theme, origin, cell * zoom, zoom, NoNode, {}, {}};
    graph.Boxes.reserve(view.Nodes.size());
    graph.Rows.reserve(view.Nodes.size());
    uint32_t columns = 1u, rows = 1u;
    for (uint32_t n = 0u; n < view.Nodes.size(); ++n) {
        graph.Rows.push_back(RowsOf(InputsOf(view, n), [&](uint16_t s) { return InTray(view, n, s); }, [&](uint16_t s) { return Shows(r, view, n, s); }));
        auto box = BoxOf(theme, cell, view.Nodes[n].Title.c_str(), NodeSockets(view, n), graph.Rows.back(), [&](uint16_t s) { return LabelOf(r, view, n, s); });
        columns = std::max(columns, uint32_t(box.W) + 2u);
        rows = std::max(rows, uint32_t(box.H) + 2u);
        graph.Boxes.push_back(box);
    }
    for (uint32_t n = 0u; n < view.Nodes.size(); ++n) {
        const auto position = view.Nodes[n].Position.value_or(uvec2{(n % 4u) * columns + 1u, (n / 4u) * rows + 1u});
        graph.Boxes[n].X = position.x;
        graph.Boxes[n].Y = position.y;
    }
    Arrange(graph.Boxes);
    if (expanded < view.Nodes.size() && graph.Rows[expanded].Tray > 0u) graph.Expanded = expanded;
    return graph;
}
// The expanded node's rectangle is taller, to fit its open tray.
ImRect RectOf(const Graph &graph, uint32_t node) {
    const auto &box = graph.Boxes[node];
    const uint32_t outputs = uint32_t(NodeSockets(graph.View, node).size()) - InputsOf(graph.View, node);
    const uint32_t height = node == graph.Expanded ? 1u + std::max({uint32_t(graph.Rows[node].OpenCount()), outputs, 1u}) : box.H;
    return {graph.Origin + ImVec2{float(box.X), float(box.Y)} * graph.Step, graph.Origin + ImVec2{float(box.X + box.W), float(box.Y + height)} * graph.Step};
}
ImVec2 RowAt(const Graph &graph, uint32_t node, bool input, uint32_t row) {
    const auto &box = graph.Boxes[node];
    return graph.Origin + ImVec2{float(box.X + (input ? 0u : box.W)), float(box.Y) + 1.5f + float(row)} * graph.Step;
}
// The sockets of a closed tray are all at the tray's chevron.
ImVec2 SocketAt(const Graph &graph, uint32_t node, uint16_t socket) {
    const auto inputs = InputsOf(graph.View, node);
    if (socket >= inputs) return RowAt(graph, node, false, socket - inputs);
    const auto &rows = graph.Rows[node];
    return RowAt(graph, node, true, node == graph.Expanded ? rows.Open[socket] : rows.Closed[socket]);
}
ImVec2 TrayAt(const Graph &graph, uint32_t node) { return RowAt(graph, node, true, graph.Rows[node].Shown); }
ImRect TrayRect(const Graph &graph, uint32_t node) {
    const float reach = graph.Theme.Sizes.SocketRadius * 1.2f * graph.Zoom + 3.f, row = graph.Step;
    const auto at = TrayAt(graph, node);
    return {{at.x - reach, at.y - row * .5f}, {at.x + reach, at.y + (float(graph.Rows[node].Tray) - .5f) * row}};
}
// The end point of link `position` of `total` on a multi-input socket.
// The links are evenly spaced and centered on the socket.
ImVec2 MultiAt(const Graph &graph, ImVec2 socket, uint32_t position, uint32_t total) {
    const float spacing = std::min(MultiSpacing, MultiHalf * 2.f / float(std::max(total, 1u))) * graph.Step;
    return {socket.x, socket.y + (float(position) - float(total - 1u) * .5f) * spacing};
}
// The index, among the socket's links, of the link a wire drag has picked up, or none.
std::optional<uint16_t> DetachedIndex(const Graph &graph, uint32_t node, uint16_t socket) {
    if (graph.DetachNode != node || SlotSocket(graph.View, node, graph.DetachSlot) != socket) return {};
    return SlotIndex(graph.View, node, graph.DetachSlot);
}
// The number of links drawn into the socket, which leaves out a picked-up link.
uint32_t ShownLinks(const Graph &graph, uint32_t node, uint16_t socket) { return LinksInto(graph.View, node, socket) - uint32_t(DetachedIndex(graph, node, socket).has_value()); }
// Where the slot's wire ends.
// On a multi-input socket, the links skip a picked-up link and leave a gap where the held wire would drop.
ImVec2 SlotAt(const Graph &graph, uint32_t node, uint16_t slot) {
    const auto &view = graph.View;
    const auto socket = SlotSocket(view, node, slot);
    const auto at = SocketAt(graph, node, socket);
    if (!NodeSockets(view, node)[socket].Multi) return at;
    uint32_t position = SlotIndex(view, node, slot), total = ShownLinks(graph, node, socket);
    if (const auto detached = DetachedIndex(graph, node, socket); detached && *detached < position) --position;
    if (graph.DropNode == node && graph.DropSocket == socket) {
        if (position >= graph.DropIndex) ++position;
        ++total;
    }
    return MultiAt(graph, at, position, total);
}

using Wire = std::array<ImVec2, 4>;
Wire WireOf(ImVec2 from, ImVec2 to, float curving) {
    const float dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const float straightening = dx > 0.f ? std::min(1.f, dy / dx * WireStraightening) : 1.f;
    const ImVec2 handle{curving * dx * straightening, 0.f};
    return {from, from + handle, to - handle, to};
}
ImVec2 NearestOnWire(const Wire &w, ImVec2 p) { return ImBezierCubicClosestPoint(w[0], w[1], w[2], w[3], p, 24); }
// Whether the wire crosses the rectangle, testing the wire as straight segments.
bool Crosses(const Wire &w, const ImRect &rect) {
    constexpr int Segments{24};
    auto a = w[0];
    for (int i = 1; i <= Segments; ++i) {
        const auto b = ImBezierCubicCalc(w[0], w[1], w[2], w[3], float(i) / float(Segments));
        const auto d = b - a;
        float enter = 0.f, leave = 1.f;
        bool inside = true;
        for (const auto [p, q] : {std::pair{-d.x, a.x - rect.Min.x}, std::pair{d.x, rect.Max.x - a.x}, std::pair{-d.y, a.y - rect.Min.y}, std::pair{d.y, rect.Max.y - a.y}}) {
            if (p == 0.f) inside &= q >= 0.f;
            else if (p < 0.f) enter = std::max(enter, q / p);
            else leave = std::min(leave, q / p);
        }
        if (inside && enter <= leave) return true;
        a = b;
    }
    return false;
}
// Calls `f(wire, link, node, slot)` for each drawn link.
void ForEachWire(const Graph &graph, auto &&f) {
    const auto &view = graph.View;
    for (uint32_t n = 0u, first = 0u; n < view.Nodes.size(); first += SlotsOf(view, n), ++n) {
        for (uint16_t slot = 0u; slot < SlotsOf(view, n); ++slot) {
            const auto link = WiresOf(view)[first + slot];
            if (link.FromNode == NoNode || (n == graph.DetachNode && slot == graph.DetachSlot)) continue;
            f(WireOf(SocketAt(graph, link.FromNode, link.FromSocket), SlotAt(graph, n, slot), graph.Theme.Sizes.WireCurving), link, n, slot);
        }
    }
}

enum class HitType : uint8_t { None,
                               Socket,
                               Parameters, // A node's closed tray
                               Node };
struct Hit {
    HitType Type{HitType::None};
    uint32_t Node{NoNode};
    uint16_t Socket{};
    uint16_t Index{}; // On a multi-input socket, the index of the link nearest the mouse
};
// Tests the expanded node first, then sockets before node bodies, with later-drawn nodes first.
Hit HitTest(const Graph &graph, ImVec2 mouse) {
    const auto &view = graph.View;
    const float reach = std::max(graph.Theme.Sizes.SocketRadius * graph.Zoom, 4.f) + 3.f;
    const auto near = [&](ImVec2 at, float half) {
        const auto d = mouse - at;
        return half > 0.f ? std::abs(d.x) <= reach && std::abs(d.y) <= half + 3.f : ImLengthSqr(d) <= reach * reach;
    };
    const auto sockets = [&](uint32_t n) -> std::optional<Hit> {
        const auto &rows = graph.Rows[n];
        const bool closed = n != graph.Expanded;
        const auto inputs = InputsOf(view, n);
        const auto all = NodeSockets(view, n);
        for (uint16_t s = 0u; s < all.size(); ++s) {
            if (s < inputs && (rows.Closed[s] == NoRow || (closed && InTray(view, n, s)))) continue;
            const auto at = SocketAt(graph, n, s);
            if (!near(at, all[s].Multi ? MultiHalf * graph.Step : 0.f)) continue;
            uint16_t index = 0u;
            if (all[s].Multi) {
                const auto links = LinksInto(view, n, s);
                for (uint16_t k = 1u; k < links; ++k) {
                    if (std::abs(MultiAt(graph, at, k, links).y - mouse.y) < std::abs(MultiAt(graph, at, index, links).y - mouse.y)) index = k;
                }
            }
            return Hit{HitType::Socket, n, s, index};
        }
        if (rows.Tray > 0u && closed && near(TrayAt(graph, n), 0.f)) return Hit{HitType::Parameters, n, 0u};
        return {};
    };
    if (graph.Expanded != NoNode) {
        if (const auto hit = sockets(graph.Expanded)) return *hit;
        if (RectOf(graph, graph.Expanded).Contains(mouse)) return {HitType::Node, graph.Expanded, 0u};
    }
    for (auto n = uint32_t(graph.Boxes.size()); n-- > 0u;) {
        if (const auto hit = sockets(n)) return *hit;
    }
    for (auto n = uint32_t(graph.Boxes.size()); n-- > 0u;) {
        if (RectOf(graph, n).Contains(mouse)) return {HitType::Node, n, 0u};
    }
    return {};
}

bool Joins(Socket held, Socket other) {
    if (held.Direction == other.Direction) return false;
    return held.Direction == SocketDirection::Out ? Links(held, other) : Links(other, held);
}
bool Accepts(const state::Registry &r, const NodeView &view, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    if (NodeSockets(view, held_node)[held_socket].Direction == NodeSockets(view, node)[socket].Direction) return false;
    if (view.Geometry) return (node != held_node || view.Nodes[node].Kind == KindOf<Subnet>) && Joins(NodeSockets(view, held_node)[held_socket], NodeSockets(view, node)[socket]);
    const auto &a = BindingOf(view, held_node, held_socket), &b = BindingOf(view, node, socket);
    const auto *input = std::get_if<SceneReference>(&a);
    return input ? ReferenceAccepts(r, *input, std::get<NetworkItem>(b)) : ReferenceAccepts(r, std::get<SceneReference>(b), std::get<NetworkItem>(a));
}
// The node's visible sockets that the held wire can link with.
// With `tray`, only its tray sockets.
std::vector<uint16_t> LinkingSockets(const state::Registry &r, const Graph &graph, uint32_t held_node, uint16_t held_socket, uint32_t node, bool tray) {
    const auto &view = graph.View;
    const auto inputs = InputsOf(view, node);
    std::vector<uint16_t> sockets;
    for (uint16_t s = 0u; s < NodeSockets(view, node).size(); ++s) {
        if (s < inputs ? graph.Rows[node].Closed[s] == NoRow || (tray && !InTray(view, node, s)) : tray) continue;
        if (Accepts(r, view, held_node, held_socket, node, s)) sockets.push_back(s);
    }
    return sockets;
}
action::nodes::Link LinkOf(const CookPlan &plan, state::Entity mesh_entity, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket, uint16_t index = UINT16_MAX) {
    const bool from_held = NodeSockets(plan, held_node)[held_socket].Direction == SocketDirection::Out;
    const auto held = plan.Nodes[held_node].Entity, other = plan.Nodes[node].Entity;
    return from_held ? action::nodes::Link{mesh_entity, held, SocketId(plan, held_node, held_socket), other, SocketId(plan, node, socket), index} : action::nodes::Link{mesh_entity, other, SocketId(plan, node, socket), held, SocketId(plan, held_node, held_socket)};
}
action::nodes::Unlink UnlinkOf(const NodeView &view, state::Entity owner, uint32_t node, uint16_t slot) {
    return {owner, view.Nodes[node].Item.Entity, SocketId(view, node, SlotSocket(view, node, slot)), SlotIndex(view, node, slot)};
}
// Links the held wire to the socket, at `index` among a multi-input socket's links.
// A wire picked up from an input is unlinked from that input in the same undo step.
// Dropping it back on that input changes nothing, or reorders it on a multi-input socket.
void Connect(const state::Registry &r, NodesPanel &panel, const NodeView &view, state::Entity owner, uint32_t node, uint16_t socket, uint16_t index = UINT16_MAX) {
    const bool detached = panel.DetachNode != NoNode;
    const auto left_node = panel.DetachNode;
    const auto left_slot = panel.DetachSlot;
    panel.DetachNode = NoNode;
    const bool returns = detached && node == left_node && socket == SlotSocket(view, left_node, left_slot);
    if (returns && !NodeSockets(view, node)[socket].Multi) return;
    std::vector<action::Action> actions;
    if (view.Geometry) {
        actions.push_back(action::MakeAction(LinkOf(*view.Geometry, owner, panel.DragNode, panel.DragSocket, node, socket, index)));
        if (detached && !returns) actions.push_back(action::MakeAction(UnlinkOf(view, owner, left_node, left_slot)));
    } else {
        const auto &a = BindingOf(view, panel.DragNode, panel.DragSocket), &b = BindingOf(view, node, socket);
        const auto *input = std::get_if<SceneReference>(&a);
        actions.push_back(action::MakeAction(input ? action::nodes::AssignReference{*input, std::get<NetworkItem>(b)} : action::nodes::AssignReference{std::get<SceneReference>(b), std::get<NetworkItem>(a)}));
        if (const auto *left = detached && !returns ? std::get_if<SceneReference>(&BindingOf(view, left_node, left_slot)) : nullptr; left && ReferenceAccepts(r, *left, {})) {
            actions.push_back(action::MakeAction(action::nodes::AssignReference{*left, {}}));
        }
    }
    action::Emit(std::move(actions));
}
void Disconnect(const state::Registry &r, const NodeView &view, state::Entity owner, uint32_t node, uint16_t slot) {
    if (view.Geometry) action::Emit(UnlinkOf(view, owner, node, slot));
    else if (const auto *reference = std::get_if<SceneReference>(&BindingOf(view, node, slot)); reference && ReferenceAccepts(r, *reference, {})) action::Emit(action::nodes::AssignReference{*reference, {}});
}
// Drops the held wire on a node.
// With one socket that can take it, links to that socket, and with several, opens the port picker.
// Returns whether the picker opened.
bool DropOnNode(const state::Registry &r, NodesPanel &panel, const Graph &graph, state::Entity mesh_entity, uint32_t node, bool tray) {
    const auto sockets = LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, node, tray);
    if (sockets.size() == 1u) Connect(r, panel, graph.View, mesh_entity, node, sockets.front());
    if (sockets.size() <= 1u) return false;
    panel.PickNode = node;
    panel.PickTray = tray;
    OpenPopup("##PickPort");
    return true;
}

// Begin moving selected nodes after the node count exceeds `after`.
void Grab(NodesPanel &panel, uint32_t after) {
    panel.Drag = NodeDrag::Grab;
    panel.DragNode = NoNode;
    panel.GrabAfter = after;
    panel.Press = vec2{GetIO().MousePos};
    panel.Moved = false;
}

std::vector<state::Entity> EntitiesOf(const CookPlan &plan, std::span<const uint32_t> indices) {
    std::vector<state::Entity> entities;
    entities.reserve(indices.size());
    for (const auto n : indices) entities.push_back(plan.Nodes[n].Entity);
    return entities;
}
std::vector<uint32_t> SelectedNodes(const state::Registry &r, const NodeView &view) {
    std::vector<uint32_t> selected;
    for (uint32_t n = 0u; n < view.Nodes.size(); ++n) {
        if (Selected(r, view, n)) selected.push_back(n);
    }
    return selected;
}

// Connect adjacent selected nodes in horizontal order, using the first compatible input.
// Skip cycles and occupied inputs unless `replace` is set.
std::vector<action::nodes::Link> MakeLinks(const state::Registry &r, state::Entity mesh_entity, const NodeView &view, std::vector<uint32_t> nodes, bool replace) {
    std::ranges::stable_sort(nodes, {}, [&](uint32_t n) { return r.get<const NodeLayout>(view.Nodes[n].Item.Entity).Position.x; });
    const auto input_first = InputsFirst(view);
    auto wires = WiresOf(view) | std::ranges::to<std::vector>();
    std::vector<action::nodes::Link> links;
    for (size_t i = 0u; i + 1u < nodes.size(); ++i) {
        auto from = nodes[i], to = nodes[i + 1u];
        if (InputsOf(view, to) == 0u || NodeSockets(view, from).size() == InputsOf(view, from)) std::swap(from, to);
        if (Upstream(input_first, std::span{wires}, from, [](uint32_t) { return true; })[to]) continue;
        const auto from_sockets = NodeSockets(view, from), to_sockets = NodeSockets(view, to);
        const auto first_link = [&](bool linked) -> std::optional<action::nodes::Link> {
            for (uint16_t input = 0u; input < InputsOf(view, to); ++input) {
                const auto current = wires[input_first[to] + input];
                if ((current.FromNode != NoNode) != linked) continue;
                for (uint16_t output = InputsOf(view, from); output < from_sockets.size(); ++output) {
                    if (Links(from_sockets[output], to_sockets[input]) && (current.FromNode != from || current.FromSocket != output)) return action::nodes::Link{mesh_entity, view.Nodes[from].Item.Entity, output, view.Nodes[to].Item.Entity, input};
                }
            }
            return {};
        };
        auto link = first_link(false);
        if (!link && replace) link = first_link(true);
        if (!link) continue;
        wires[input_first[to] + link->ToSocket] = {from, link->FromSocket};
        link->FromSocket = SocketId(view, from, link->FromSocket);
        link->ToSocket = SocketId(view, to, link->ToSocket);
        links.push_back(*link);
    }
    return links;
}

// The link a dragged node would be inserted into, and the cells that link's reader moves right to make room.
struct Insert {
    uint32_t ToNode;
    uint16_t ToSlot;
    uint32_t Shift;
};
// Of the links that cross the node's box and that it can be inserted into, the one passing nearest its main input socket.
std::optional<Insert> InsertTarget(const Graph &graph, uint32_t node) {
    const auto &view = graph.View;
    const auto input = view.Geometry ? MainSocket(view.Nodes[node].Kind, SocketDirection::In) : std::nullopt;
    if (!input) return {};
    const auto rect = RectOf(graph, node);
    const auto entry = SocketAt(graph, node, *input);
    std::optional<Insert> insert;
    float nearest = FLT_MAX;
    ForEachWire(graph, [&](const Wire &wire, NodeLink, uint32_t to_node, uint16_t slot) {
        if (!Crosses(wire, rect) || !Inserts(*view.Geometry, node, to_node, slot)) return;
        const float distance = ImLengthSqr(NearestOnWire(wire, entry) - entry);
        if (distance >= nearest) return;
        nearest = distance;
        const auto &box = graph.Boxes[node], &reader = graph.Boxes[to_node];
        const uint32_t clear = box.X + box.W + 1u;
        insert = Insert{to_node, slot, clear > reader.X ? clear - reader.X : 0u};
    });
    return insert;
}

void Interact(const state::Registry &r, NodesPanel &panel, state::Entity mesh_entity, const Graph &graph, const std::vector<uint32_t> &selected, const Hit &hit, const std::optional<Insert> &insert) {
    const auto &io = GetIO();
    const auto &view = graph.View;
    const auto &nodes = view.Nodes;
    const auto active_node = ActiveIndex(r, view);
    const auto mouse = io.MousePos;
    const bool mouse_moved = io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f;
    const auto select = [&](const std::vector<uint32_t> &selection, uint32_t active) {
        action::Emit(action::nodes::SelectItems{view.Network, active == NoNode ? NetworkItem{} : nodes[active].Item, ItemsOf(view, selection)});
    };
    const auto open_add = [&](bool linking) {
        const auto cell = (mouse - graph.Origin) / graph.Step;
        panel.AddAt = {std::max(std::floor(cell.x), 0.f), std::max(std::floor(cell.y), 0.f)};
        panel.Linking = linking;
        OpenPopup("##AddNode");
    };
    // Moves the selected nodes together to the cells nearest the mouse where they fit among the other nodes.
    const auto follow = [&](bool first) {
        const auto &held = graph.Boxes[panel.DragNode];
        const int64_t wanted_x = int64_t(panel.MoveStart.x) + std::lround((mouse.x - panel.Press.x) / graph.Step), wanted_y = int64_t(panel.MoveStart.y) + std::lround((mouse.y - panel.Press.y) / graph.Step);
        std::vector<bool> others(nodes.size(), true);
        for (const auto n : selected) others[n] = false;
        const auto [dx, dy] = Place(graph.Boxes, others, selected, wanted_x - int64_t(held.X), wanted_y - int64_t(held.Y));
        if (first || dx != 0 || dy != 0) {
            action::nodes::MoveNodes move{EditorNetwork(r, panel), {}};
            for (const auto n : selected) {
                const auto &box = graph.Boxes[n];
                move.Positions.push_back({nodes[n].Item, {uint32_t(int64_t(box.X) + dx), uint32_t(int64_t(box.Y) + dy)}});
            }
            action::Emit(std::move(move), action::Phase::Stage);
        }
    };
    const auto end_move = [&](bool commit) {
        if (panel.Moved && commit) {
            if (insert) {
                const auto socket = SlotSocket(view, insert->ToNode, insert->ToSlot);
                action::Emit(action::nodes::InsertNode{mesh_entity, nodes[panel.DragNode].Item.Entity, nodes[insert->ToNode].Item.Entity, SocketId(view, insert->ToNode, socket), insert->Shift, SlotIndex(view, insert->ToNode, insert->ToSlot)}, action::Phase::Stage);
            }
            action::Commit();
        } else if (panel.Moved) {
            action::Cancel();
        }
        panel.Drag = NodeDrag::None;
    };
    if (panel.Drag == NodeDrag::Grab) {
        if (panel.DragNode == NoNode) {
            if (nodes.size() <= panel.GrabAfter) {
                panel.Drag = NodeDrag::None;
                return;
            }
            panel.DragNode = active_node;
            panel.MoveStart = {graph.Boxes[panel.DragNode].X, graph.Boxes[panel.DragNode].Y};
        }
        if (IsMouseClicked(ImGuiMouseButton_Left)) end_move(true);
        else if (IsKeyPressed(ImGuiKey_Escape) || IsMouseClicked(ImGuiMouseButton_Right)) end_move(false);
        else if (!panel.Moved || mouse_moved) {
            const bool first = !panel.Moved;
            panel.Moved = true;
            follow(first);
        }
        return;
    }
    if (panel.Drag == NodeDrag::None && panel.Hovered) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            panel.Press = vec2{mouse};
            if (hit.Type == HitType::Socket) {
                // Dragging from a linked input picks up that link, and the drag then holds the link's output end.
                panel.Drag = NodeDrag::Wire;
                if (hit.Socket < InputsOf(view, hit.Node) && LinksInto(view, hit.Node, hit.Socket) > 0u) {
                    const auto slot = SlotOf(view, hit.Node, hit.Socket, hit.Index);
                    const auto link = WiresOf(view)[InputsFirst(view)[hit.Node] + slot];
                    panel.DetachNode = hit.Node;
                    panel.DetachSlot = slot;
                    panel.DragNode = link.FromNode;
                    panel.DragSocket = link.FromSocket;
                } else {
                    panel.DetachNode = NoNode;
                    panel.DragNode = hit.Node;
                    panel.DragSocket = hit.Socket;
                }
            } else if (hit.Type == HitType::Node || hit.Type == HitType::Parameters) {
                if (IsMouseDoubleClicked(ImGuiMouseButton_Left) && EnterNode(r, panel, nodes[hit.Node].Item)) {
                    return;
                }
                const bool is_selected = Selected(r, view, hit.Node), active = hit.Node == active_node;
                if (io.KeyShift && is_selected && active) {
                    auto remaining = selected;
                    std::erase(remaining, hit.Node);
                    select(remaining, NoNode);
                } else {
                    if (!io.KeyShift && !is_selected) {
                        select({hit.Node}, hit.Node);
                    } else if (!active) {
                        auto extended = selected;
                        if (!is_selected) extended.insert(std::ranges::lower_bound(extended, hit.Node), hit.Node);
                        select(extended, hit.Node);
                    }
                    panel.Drag = NodeDrag::Move;
                    panel.DragNode = hit.Node;
                    panel.Moved = false;
                }
            } else {
                panel.Drag = NodeDrag::Box;
            }
        } else if (IsMouseClicked(ImGuiMouseButton_Right)) {
            if (!view.Geometry && (hit.Type == HitType::Node || hit.Type == HitType::Socket)) {
                panel.ContextNode = hit.Node;
                panel.ContextSocket = hit.Type == HitType::Socket ? hit.Socket : UINT16_MAX;
                OpenPopup("##NodeMenu");
            } else open_add(false);
        }
    }
    if (panel.Drag == NodeDrag::None && panel.Hovered) {
        constexpr auto Key{ImGuiInputFlags_RouteGlobal};
        if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, Key)) {
            open_add(false);
        } else if (Shortcut(ImGuiKey_X, Key) || Shortcut(ImGuiKey_Delete, Key) || Shortcut(ImGuiKey_Backspace, Key)) {
            if (view.Geometry && !selected.empty()) action::Emit(action::nodes::DeleteNodes{mesh_entity, EntitiesOf(*view.Geometry, selected)});
        } else if (const bool replace = Shortcut(ImGuiMod_Shift | ImGuiKey_J, Key); replace || Shortcut(ImGuiKey_J, Key)) {
            if (view.Geometry) panel.PendingLinks = MakeLinks(r, mesh_entity, view, selected, replace);
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, Key)) {
            if (view.Geometry && !selected.empty()) {
                action::Emit(action::nodes::DuplicateNodes{mesh_entity, EntitiesOf(*view.Geometry, selected)});
                Grab(panel, uint32_t(nodes.size()));
            }
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_G, Key)) {
            if (view.Geometry) {
                auto grouped = EntitiesOf(*view.Geometry, selected);
                std::erase(grouped, panel.Canvas.Network);
                if (!grouped.empty()) action::Emit(action::nodes::GroupNodes{mesh_entity, std::move(grouped)});
            } else if (!selected.empty()) action::Emit(action::nodes::GroupItems{view.Network, ItemsOf(view, selected)});
        } else if (Shortcut(ImGuiKey_G, Key)) {
            if (active_node != NoNode && Selected(r, view, active_node)) Grab(panel, uint32_t(nodes.size()) - 1u);
        } else if (Shortcut(ImGuiKey_A, Key)) {
            std::vector<uint32_t> all(nodes.size());
            std::ranges::iota(all, 0u);
            if (selected != all) select(all, NoNode);
        } else if (Shortcut(ImGuiMod_Alt | ImGuiKey_A, Key)) {
            if (!selected.empty()) select({}, NoNode);
        }
    }
    if (panel.Drag == NodeDrag::Move) {
        if (!IsMouseDown(ImGuiMouseButton_Left)) end_move(true);
        else if (panel.Moved && IsKeyPressed(ImGuiKey_Escape)) end_move(false);
        else if (IsMouseDragging(ImGuiMouseButton_Left)) {
            const bool first = !panel.Moved;
            if (first && panel.DragNode == active_node) {
                panel.Moved = true;
                panel.MoveStart = {graph.Boxes[panel.DragNode].X, graph.Boxes[panel.DragNode].Y};
            }
            if (!panel.Moved) panel.Drag = NodeDrag::None;
            else if (first || mouse_moved) follow(first);
        }
    }
    if (panel.Drag == NodeDrag::Wire && IsKeyPressed(ImGuiKey_Escape)) {
        panel.Drag = NodeDrag::None;
        panel.DetachNode = NoNode;
    }
    if (panel.Drag == NodeDrag::Wire && !IsMouseDown(ImGuiMouseButton_Left)) {
        bool picking = false;
        if (hit.Type == HitType::Socket) {
            const bool dropping = graph.DropNode == hit.Node && graph.DropSocket == hit.Socket;
            if (Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket)) Connect(r, panel, view, mesh_entity, hit.Node, hit.Socket, dropping ? graph.DropIndex : UINT16_MAX);
        } else if (hit.Type == HitType::Parameters || hit.Type == HitType::Node) picking = DropOnNode(r, panel, graph, mesh_entity, hit.Node, hit.Type == HitType::Parameters);
        else if (panel.DetachNode != NoNode) Disconnect(r, view, mesh_entity, panel.DetachNode, panel.DetachSlot);
        else if (panel.Hovered && view.Geometry) open_add(true);
        if (!picking) panel.DetachNode = NoNode;
        panel.Drag = NodeDrag::None;
    }
    if (panel.Drag == NodeDrag::Box && !IsMouseDown(ImGuiMouseButton_Left)) {
        const auto press = Im(panel.Press);
        if (ImLengthSqr(mouse - press) >= io.MouseDragThreshold * io.MouseDragThreshold) {
            const ImRect box{ImMin(press, mouse), ImMax(press, mouse)};
            std::vector<uint32_t> banded;
            for (uint32_t n = 0u; n < nodes.size(); ++n) {
                if (box.Overlaps(RectOf(graph, n)) || (io.KeyShift && std::ranges::binary_search(selected, n))) banded.push_back(n);
            }
            if (banded != selected) select(banded, NoNode);
        } else if (!io.KeyShift && !selected.empty()) {
            select({}, NoNode);
        }
        panel.Drag = NodeDrag::None;
    }
}

char Lower(char c) { return char(std::tolower(static_cast<unsigned char>(c))); }
bool Matches(std::string_view label, std::string_view text) { return text.empty() || !std::ranges::search(label, text, {}, Lower, Lower).empty(); }
// An Add menu search result: a node kind, optionally with a starting operation.
struct AddEntry {
    uint16_t Kind;
    std::optional<uint32_t> Operation;
    const char *Label;
};
template<typename K> constexpr size_t OperationCount = [] {
    if constexpr (Operated<K>) return field::EnumCount<decltype(K::Operation)>;
    else return size_t{0};
}();
// Every kind, each followed by one entry per operation.
constexpr auto AddEntries{[]<typename... Ks>(TypeList<Ks...>) {
    std::array<AddEntry, (sizeof...(Ks) + ... + OperationCount<Ks>)> entries{};
    size_t n = 0;
    const auto add = [&]<typename K> {
        entries[n++] = {KindOf<K>, {}, KindTitles[KindOf<K>]};
        if constexpr (Operated<K>) {
            for (const auto op : field::EnumValues<decltype(K::Operation)>) entries[n++] = {KindOf<K>, uint32_t(op), field::EnumLabel(op)};
        }
    };
    (add.template operator()<Ks>(), ...);
    return entries;
}(NodeKinds{})};
// Search labels case-insensitively, placing prefix matches first.
std::vector<AddEntry> SearchEntries(std::string_view text) {
    auto entries = AddEntries | std::views::filter([&](const AddEntry &entry) { return Matches(entry.Label, text); }) | std::ranges::to<std::vector>();
    std::ranges::stable_partition(entries, [&](const AddEntry &entry) { return std::ranges::starts_with(std::string_view{entry.Label}, text, {}, Lower, Lower); });
    return entries;
}

int MoveHighlight(ImGuiInputTextCallbackData *data) {
    *static_cast<int *>(data->UserData) += data->EventKey == ImGuiKey_DownArrow ? 1 : -1;
    return 0;
}

struct LinkingEntry {
    uint16_t Kind, Socket;
    const char *SocketName;
    std::string Label;
};
std::vector<LinkingEntry> LinkingEntries(Socket held) {
    std::vector<LinkingEntry> entries;
    for (uint16_t kind = 0u; kind < KindCount; ++kind) {
        const auto sockets = KindSockets(kind);
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const auto *name = LabelOf(sockets[s]);
            if (Joins(held, sockets[s])) entries.push_back({kind, s, name, std::string{KindTitles[kind]} + " " + name});
        }
    }
    return entries;
}
// The cell offset of the socket from the top left of a newly added node of the kind.
ImVec2 NewSocketCell(const NodeTheme &theme, float cell, uint16_t kind, uint16_t socket) {
    const auto sockets = KindSockets(kind);
    const auto inputs = InputCount[kind];
    const auto rows = RowsOf(inputs, [&](uint16_t s) { return InTray(kind, s); }, [&](uint16_t s) { return s == socket || kind != KindOf<FloatMath> || s >= OperandCount || OperandLabels(FloatMath{}.Operation)[s]; });
    if (socket < inputs) return {0.f, 1.f + float(rows.Closed[socket])};
    return {float(BoxOf(theme, cell, KindTitles[kind], sockets, rows, [&](uint16_t s) { return LabelOf(sockets[s]); }).W), 1.f + float(socket - inputs)};
}
constexpr int ResultRows{10};
// row(i, highlighted) returns whether row i was clicked.
// Returns the clicked row, or the highlighted row on Enter, and closes the popup.
std::optional<int> DrawResults(int count, float width, int &highlight, int previous, bool enter, auto &&row) {
    highlight = std::max(0, std::min(highlight, count - 1));
    const float rows = float(std::min(count, ResultRows));
    BeginChild("##results", {width, rows > 0.f ? rows * GetTextLineHeightWithSpacing() : 1.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    std::optional<int> picked;
    for (int i = 0; i < count; ++i) {
        if (row(i, i == highlight)) picked = i;
        if (i == highlight && highlight != previous) SetScrollHereY();
    }
    EndChild();
    if (enter && count > 0) picked = highlight;
    if (picked) CloseCurrentPopup();
    return picked;
}

void DrawAddMenu(NodesPanel &panel, const NodeTheme &theme, float cell, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    if (!BeginPopup("##AddNode")) return;
    const auto &nodes = plan.Nodes;
    const bool linking = panel.Linking && panel.DragNode < nodes.size();
    const auto add = [&](uint16_t kind, std::optional<uint32_t> operation = {}) {
        if (!interactive) return;
        action::Emit(action::nodes::AddNode{mesh_entity, kind, panel.AddAt, panel.Canvas.Network == mesh_entity ? state::Null : panel.Canvas.Network, operation});
        Grab(panel, uint32_t(nodes.size()));
    };
    if (IsWindowAppearing()) {
        panel.Search = {};
        panel.SearchHighlight = 0;
        SetKeyboardFocusHere();
    }
    const auto highlight = panel.SearchHighlight;
    SetNextItemWidth(GetFontSize() * 14.f);
    const bool enter = InputTextWithHint("##Search", "Search", panel.Search.data(), panel.Search.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory, MoveHighlight, &panel.SearchHighlight);
    if (IsItemEdited()) panel.SearchHighlight = 0;
    const std::string_view text{panel.Search.data()};
    if (linking) {
        auto entries = LinkingEntries(NodeSockets(plan, panel.DragNode)[panel.DragSocket]);
        std::erase_if(entries, [&](const LinkingEntry &entry) { return !Matches(entry.Label, text); });
        float width = GetFontSize() * 14.f;
        for (const auto &entry : entries) width = std::max(width, CalcTextSize(entry.Label.c_str()).x + GetFontSize() * 2.f);
        const auto picked = DrawResults(int(entries.size()), width, panel.SearchHighlight, highlight, enter, [&](int i, bool highlighted) {
            PushID(i);
            const auto at = GetCursorScreenPos();
            const bool clicked = Selectable("##entry", highlighted);
            PopID();
            SetCursorScreenPos(at);
            TextDisabled("%s >", KindTitles[entries[i].Kind]);
            SameLine();
            TextUnformatted(entries[i].SocketName);
            return clicked;
        });
        if (picked && interactive) {
            const auto &entry = entries[*picked];
            const auto socket = NewSocketCell(theme, cell, entry.Kind, entry.Socket);
            const uvec2 at{std::max(float(panel.AddAt.x) - socket.x, 0.f), std::max(float(panel.AddAt.y) - socket.y, 0.f)};
            action::Emit(action::nodes::AddNode{mesh_entity, entry.Kind, at, panel.Canvas.Network == mesh_entity ? state::Null : panel.Canvas.Network}, action::Phase::Stage);
            panel.LinkAfter = uint32_t(nodes.size());
            panel.LinkSocket = entry.Socket;
        }
    } else if (text.empty()) {
        for (uint8_t group = 0u; group < uint8_t(std::size(GroupNames)); ++group) {
            if (!BeginMenu(GroupNames[group])) continue;
            for (uint16_t kind = 0u; kind < KindCount; ++kind) {
                if (KindGroups[kind] == KindGroup(group) && MenuItem(KindTitles[kind])) add(kind);
            }
            EndMenu();
        }
        if (BeginMenu("Macros")) {
            for (uint16_t macro = 0u; macro < Macros.size(); ++macro) {
                if (MenuItem(Macros[macro].Name) && interactive) {
                    action::Emit(action::nodes::AddMacro{mesh_entity, macro, panel.AddAt, panel.Canvas.Network == mesh_entity ? state::Null : panel.Canvas.Network});
                    Grab(panel, uint32_t(nodes.size()));
                }
            }
            EndMenu();
        }
    } else {
        const auto found = SearchEntries(text);
        const auto picked = DrawResults(int(found.size()), GetFontSize() * 22.f, panel.SearchHighlight, highlight, enter, [&](int i, bool highlighted) {
            const auto &entry = found[i];
            PushID(i);
            const auto at = GetCursorScreenPos();
            const bool clicked = Selectable("##entry", highlighted);
            PopID();
            SetCursorScreenPos(at);
            if (entry.Operation) TextDisabled("%s > %s >", GroupNames[uint8_t(KindGroups[entry.Kind])], KindTitles[entry.Kind]);
            else TextDisabled("%s >", GroupNames[uint8_t(KindGroups[entry.Kind])]);
            SameLine();
            TextUnformatted(entry.Label);
            return clicked;
        });
        if (picked) add(found[*picked].Kind, found[*picked].Operation);
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void DrawPortPicker(const state::Registry &r, NodesPanel &panel, state::Entity mesh_entity, const Graph &graph, bool interactive) {
    if (panel.PickNode == NoNode) return;
    const auto &view = graph.View;
    if (!BeginPopup("##PickPort")) {
        // Closing the picker without a choice leaves a picked-up link where it was.
        panel.PickNode = NoNode;
        if (panel.Drag != NodeDrag::Wire) panel.DetachNode = NoNode;
        return;
    }
    const auto node = panel.PickNode;
    if (node < view.Nodes.size() && panel.DragNode < view.Nodes.size()) {
        for (const auto s : LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, node, panel.PickTray)) {
            if (MenuItem(LabelOf(r, view, node, s)) && interactive) Connect(r, panel, view, mesh_entity, node, s);
        }
    } else {
        CloseCurrentPopup();
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void RecordHover(const state::Registry &r, const NodesPanel &panel, const NodeView &view, const Hit &hit) {
    if (hit.Type == HitType::None) return;
    const auto &node = view.Nodes[hit.Node];
    const auto entity = node.Item.Entity;
    const auto kind = node.Kind;
    if (hit.Type == HitType::Node) {
        const auto *error = r.try_get<const NodeError>(entity);
        ui::Hovered = {.Label = view.Geometry ? OperationTitle(r, entity, kind) : node.Type, .Path = node.Type, .Type = node.Type, .Help = view.Geometry ? KindHelp[kind] : nullptr, .Status = error ? FaultTexts[uint8_t(error->Fault)] : nullptr};
    } else if (hit.Type == HitType::Parameters) {
        ui::Hovered = {.Label = "Parameters", .Path = node.Type, .Help = "Hover to show the sockets of the node's value parameters. Drop a wire here to link it to one of them."};
    } else {
        const auto socket = NodeSockets(view, hit.Node)[hit.Socket];
        if (!view.Geometry) SetTooltip("%s: %s", LabelOf(socket), ReferenceDescription(r, BindingOf(view, hit.Node, hit.Socket)).c_str());
        const bool held = panel.DragNode == hit.Node && panel.DragSocket == hit.Socket;
        const bool dimmed = panel.Drag == NodeDrag::Wire && !held && !Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket);
        const bool linked = socket.Direction == SocketDirection::In && IsLinked(view, hit.Node, hit.Socket);
        ui::Hovered = {
            .Label = view.Geometry ? LabelOf(r, view, hit.Node, hit.Socket) : "Reference",
            .Path = node.Type,
            .Type = SocketTypeText(socket),
            // A linked input's parameter value is unused, so its help says how to move the link.
            .Help = !view.Geometry ? "Reference to a document item" : socket.Multi ? "Accepts any number of links. They apply in order from top to bottom. Drag a link along the socket to reorder it." :
                linked                                                             ? "Drag to move this link to another input. Drop it on empty canvas to unlink." :
                kind == KindOf<Subnet>                                             ? "Named subnet interface port" :
                                                                                     SocketHelps[SocketFirst[kind] + hit.Socket],
            .Status = dimmed ? "Cannot link with the held wire" : linked ? "Linked" :
                                                                           nullptr,
        };
    }
}

void DrawCanvas(const state::Registry &r, FrameState &frame, NodesPanel &panel, const NodeTheme &theme, state::Entity mesh_entity, const NodeView &view, bool interactive) {
    const auto &io = GetIO();
    const auto &nodes = view.Nodes;
    if ((panel.Drag == NodeDrag::Move || panel.Drag == NodeDrag::Wire || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode)) && panel.DragNode >= nodes.size()) panel.Drag = NodeDrag::None;
    if (panel.DetachNode != NoNode && (panel.DetachNode >= nodes.size() || panel.DetachSlot >= SlotsOf(view, panel.DetachNode))) {
        panel.DetachNode = NoNode;
        if (panel.Drag == NodeDrag::Wire) panel.Drag = NodeDrag::None;
    }
    const auto origin = GetCursorScreenPos();
    const auto size = GetContentRegionAvail();
    InvisibleButton("##canvas", {std::max(size.x, 1.f), std::max(size.y, 1.f)}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    const auto end = GetItemRectMax();
    const auto mouse = io.MousePos;
    panel.Hovered = IsItemHovered() || (IsItemActive() && ImRect{origin, end}.Contains(mouse));
    SetItemKeyOwner(ImGuiKey_MouseWheelY);
    auto &canvas = panel.Canvas;
    if (interactive) {
        if (panel.Hovered) {
            const auto wheel = std::exchange(frame.PreciseWheelDelta, vec2{0});
            const auto magnification = std::exchange(frame.Magnification, 0.f);
            const bool pan = frame.PreciseScrolling && !io.KeyCtrl && !io.KeySuper;
            if (pan) canvas.Offset = vec2{Im(canvas.Offset) + Im(wheel) * TrackpadPoints};
            const float steps = pan ? 0.f : wheel.y * (frame.PreciseScrolling ? TrackpadPoints / TrackpadStepPoints : 1.f);
            if (const auto zoom = std::clamp(canvas.Zoom * std::pow(ZoomStep, steps) * (1.f + magnification), MinZoom, MaxZoom); zoom != canvas.Zoom) {
                const auto at = (mouse - origin - Im(canvas.Offset)) / canvas.Zoom;
                canvas.Offset = vec2{mouse - origin - at * zoom};
                canvas.Zoom = zoom;
            }
        }
        if (panel.Drag == NodeDrag::None && panel.Hovered && IsMouseClicked(ImGuiMouseButton_Middle)) panel.Drag = NodeDrag::Pan;
        if (panel.Drag == NodeDrag::Pan) {
            if (IsMouseDown(ImGuiMouseButton_Middle)) canvas.Offset = vec2{Im(canvas.Offset) + io.MouseDelta};
            else panel.Drag = NodeDrag::None;
        }
    }
    // Clamp the offset so nothing left of or above the margin shows.
    canvas.Offset = {std::min(canvas.Offset.x, 0.f), std::min(canvas.Offset.y, 0.f)};
    const float cell = CellOf(theme);
    const float margin = CanvasMargin * cell * canvas.Zoom;
    auto graph = GraphOf(r, view, theme, origin + Im(canvas.Offset) + ImVec2{margin, margin}, cell, canvas.Zoom, panel.Expanded);
    const bool wiring = panel.Drag == NodeDrag::Wire;
    if (wiring || panel.PickNode != NoNode) {
        graph.DetachNode = panel.DetachNode;
        graph.DetachSlot = panel.DetachSlot;
    }
    const auto hit = panel.Hovered ? HitTest(graph, mouse) : Hit{};
    // While the held wire is over a multi-input socket it can link with, the mouse height sets the index it would be inserted at.
    std::optional<ImVec2> drop_at;
    if (wiring && hit.Type == HitType::Socket && NodeSockets(view, hit.Node)[hit.Socket].Multi && Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket)) {
        const auto at = SocketAt(graph, hit.Node, hit.Socket);
        const auto shown = ShownLinks(graph, hit.Node, hit.Socket);
        graph.DropNode = hit.Node;
        graph.DropSocket = hit.Socket;
        for (uint32_t k = 0u; k < shown; ++k) graph.DropIndex += MultiAt(graph, at, k, shown).y < mouse.y;
        drop_at = MultiAt(graph, at, graph.DropIndex, shown + 1u);
    }
    if (panel.Drag == NodeDrag::None || wiring) RecordHover(r, panel, view, hit);
    const auto selected = SelectedNodes(r, view);
    const bool holding = (panel.Drag == NodeDrag::Move && panel.Moved) || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode);
    const auto insert = holding && selected == std::vector{panel.DragNode} ? InsertTarget(graph, panel.DragNode) : std::nullopt;
    if (interactive) Interact(r, panel, mesh_entity, graph, selected, hit, insert);
    const bool stays_open = graph.Expanded != NoNode && (TrayRect(graph, graph.Expanded).Contains(mouse) || (wiring && panel.DragNode == graph.Expanded) || panel.PickNode == graph.Expanded);
    if (hit.Type == HitType::Parameters) panel.Expanded = hit.Node;
    else if (!stays_open) panel.Expanded = NoNode;
    const bool idle = panel.Drag == NodeDrag::None && !wiring;
    const auto target = panel.Drag == NodeDrag::None ? std::nullopt : insert;

    auto &draw = *GetWindowDrawList();
    const auto zoom = canvas.Zoom;
    const auto &colors = theme.Colors;
    draw.PushClipRect(origin, end, true);
    draw.AddRectFilled(origin, end, U32(colors.Background));
    const auto grid = U32(colors.Grid), coarse_grid = U32(colors.CoarseGrid);
    const auto coarse = int32_t(std::max<uint8_t>(theme.Sizes.CoarseGridCells, 1u));
    const auto grid_min = ImMax(origin, graph.Origin);
    for (auto i = std::max(0, int32_t(std::floor((origin.x - graph.Origin.x) / graph.Step))); graph.Origin.x + float(i) * graph.Step < end.x; ++i) {
        const float x = graph.Origin.x + float(i) * graph.Step;
        if (x >= origin.x) draw.AddLine({x, grid_min.y}, {x, end.y}, i % coarse == 0 ? coarse_grid : grid);
    }
    for (auto i = std::max(0, int32_t(std::floor((origin.y - graph.Origin.y) / graph.Step))); graph.Origin.y + float(i) * graph.Step < end.y; ++i) {
        const float y = graph.Origin.y + float(i) * graph.Step;
        if (y >= origin.y) draw.AddLine({grid_min.x, y}, {end.x, y}, i % coarse == 0 ? coarse_grid : grid);
    }
    const float thickness = std::max(theme.Sizes.WireWidth * zoom, 1.f);
    // Highlight a wire when a dragged node would be inserted into it, when a drag from the hovered input would pick it up, or when it starts at the hovered output.
    const bool over_input = idle && hit.Type == HitType::Socket && hit.Socket < InputsOf(view, hit.Node);
    const bool over_output = idle && hit.Type == HitType::Socket && !over_input;
    const auto hovered_slot = over_input ? SlotOf(view, hit.Node, hit.Socket, hit.Index) : uint16_t{};
    ForEachWire(graph, [&](const Wire &wire, NodeLink link, uint32_t node, uint16_t slot) {
        const bool lit = (target && target->ToNode == node && target->ToSlot == slot) || (over_input && hit.Node == node && hovered_slot == slot) || (over_output && link == NodeLink{hit.Node, hit.Socket});
        const auto color = lit ? colors.WireHighlight : SocketColor(colors, NodeSockets(view, link.FromNode)[link.FromSocket]);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(color), lit ? thickness * 1.5f : thickness);
    });
    for (const auto [from, to] : view.Relationships) {
        const auto a = RectOf(graph, from), b = RectOf(graph, to);
        const auto wire = WireOf({a.Max.x, a.Min.y + graph.Step * .5f}, {b.Min.x, b.Min.y + graph.Step * .5f}, theme.Sizes.WireCurving);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(colors.MixedSocket), thickness);
    }
    PushFont(nullptr, GetStyle().FontSizeBase * zoom);
    const auto font_size = GetFontSize();
    const float inset = (theme.Sizes.SocketRadius + theme.Sizes.LabelSpacing) * zoom, spacing = theme.Sizes.LabelSpacing * zoom, label_reach = float(theme.Sizes.MaxLabelCells) * graph.Step;
    const auto draw_label = [&](ImVec2 min, ImVec2 max, const char *text, vec4 color, bool right) {
        if (const float width = CalcTextSize(text).x; right && width <= max.x - min.x) min.x = max.x - width;
        PushStyleColor(ImGuiCol_Text, ImVec4{color.x, color.y, color.z, color.w});
        RenderTextEllipsis(&draw, min, max, max.x, text, nullptr, nullptr);
        PopStyleColor();
    };
    const auto active_node = ActiveIndex(r, view);
    const auto draw_node = [&](uint32_t n) {
        const auto &node = nodes[n];
        const auto entity = node.Item.Entity;
        const auto kind = node.Kind;
        const auto [min, max] = RectOf(graph, n);
        const auto &rows = graph.Rows[n];
        const bool expanded = n == graph.Expanded;
        const float rounding = theme.Sizes.Rounding * zoom, pad = theme.Sizes.TitlePadding * zoom;
        draw.AddRectFilled(min, max, U32(expanded ? vec4{colors.Node.x, colors.Node.y, colors.Node.z, 1.f} : colors.Node), rounding);
        draw.AddRectFilled(min, {max.x, min.y + graph.Step}, U32(n == view.DisplayNode ? colors.DisplayedTitle : colors.*GroupColors[view.Geometry ? uint8_t(KindGroups[kind]) : uint8_t(KindGroup::Sources)]), rounding, ImDrawFlags_RoundTop);
        draw_label({min.x + pad, min.y + (graph.Step - font_size) * .5f}, {max.x - pad, min.y + (graph.Step + font_size) * .5f}, node.Title.c_str(), colors.Title, false);
        const auto sockets = NodeSockets(view, n);
        const auto inputs = InputsOf(view, n);
        const bool erring = r.all_of<NodeError>(entity);
        const bool active = n == active_node, is_selected = Selected(r, view, n);
        const auto border = active ? colors.Active : is_selected ? colors.Selected :
            erring                                               ? colors.Error :
                                                                   colors.Border;
        draw.AddRect(min, max, U32(border), rounding, erring || active || is_selected ? theme.Sizes.HighlightBorderWidth : theme.Sizes.BorderWidth, ImDrawFlags_StrokeCenter);
        if (erring && (active || is_selected)) draw.AddRect(min + ImVec2{3.f, 3.f}, max - ImVec2{3.f, 3.f}, U32(colors.Error), rounding, 1.f);
        if (expanded) {
            const auto tray = TrayRect(graph, n);
            draw.AddRectFilled(tray.Min, tray.Max, U32(colors.Tray), rounding);
            draw.AddRect(tray.Min, tray.Max, U32(colors.Outline), rounding, theme.Sizes.OutlineWidth);
        }
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const bool parameter = s < inputs && InTray(view, n, s);
            if (s < inputs && (rows.Closed[s] == NoRow || (parameter && !expanded))) continue;
            const auto at = SocketAt(graph, n, s);
            const bool held = wiring && panel.DragNode == n && panel.DragSocket == s;
            const bool dimmed = wiring && !held && !Accepts(r, view, panel.DragNode, panel.DragSocket, n, s);
            const bool hovered = hit.Type == HitType::Socket && hit.Node == n && hit.Socket == s && !dimmed;
            DrawSocket(draw, theme, at, theme.Sizes.SocketRadius * zoom, MultiHalf * graph.Step, sockets[s], dimmed ? colors.Dimmed : SocketColor(colors, sockets[s]), hovered);
            const auto color = dimmed ? colors.DimmedLabel : parameter ? colors.ParameterLabel :
                                                                         colors.Label;
            const float top = at.y - font_size * .5f, bottom = at.y + font_size * .5f;
            if (s < inputs) draw_label({at.x + inset + (parameter ? theme.Sizes.ParameterIndent * zoom : 0.f), top}, {std::min(at.x + label_reach, max.x) - spacing, bottom}, LabelOf(r, view, n, s), color, false);
            else draw_label({std::max(at.x - label_reach, min.x) + spacing, top}, {at.x - inset, bottom}, LabelOf(r, view, n, s), color, true);
        }
        if (rows.Tray > 0u && !expanded) {
            const auto at = TrayAt(graph, n);
            std::string label;
            for (uint16_t s = 0u; s < inputs; ++s) {
                if (InTray(view, n, s) && IsLinked(view, n, s)) label += std::string{label.empty() ? "" : ", "} + LabelOf(r, view, n, s);
            }
            const bool linked = !label.empty();
            const bool dimmed = wiring && LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, n, true).empty();
            const bool hovered = hit.Type == HitType::Parameters && hit.Node == n && !dimmed;
            const auto color = dimmed ? colors.Dimmed : hovered ? colors.Hovered :
                linked                                          ? colors.ChevronLinked :
                                                                  colors.Chevron;
            const float chevron = theme.Sizes.SocketRadius * zoom;
            const ImVec2 a{at.x - chevron * .7f, at.y - chevron}, b{at.x - chevron * .7f, at.y + chevron}, c{at.x + chevron * .9f, at.y};
            draw.AddTriangleFilled(a, c, b, U32(color));
            draw.AddTriangle(a, c, b, U32(colors.Outline), theme.Sizes.OutlineWidth);
            if (!linked) label = "Parameters";
            draw_label({at.x + inset, at.y - font_size * .5f}, {max.x - spacing, at.y + font_size * .5f}, label.c_str(), color, false);
        }
    };
    for (uint32_t n = 0u; n < nodes.size(); ++n) {
        if (n != graph.Expanded) draw_node(n);
    }
    if (graph.Expanded != NoNode) draw_node(graph.Expanded);
    PopFont();
    if (wiring) {
        const auto socket = SocketAt(graph, panel.DragNode, panel.DragSocket);
        const auto held = NodeSockets(view, panel.DragNode)[panel.DragSocket];
        const auto wire = held.Direction == SocketDirection::Out ? WireOf(socket, drop_at.value_or(mouse), theme.Sizes.WireCurving) : WireOf(mouse, socket, theme.Sizes.WireCurving);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(SocketColor(colors, held)), thickness);
    }
    if (panel.Drag == NodeDrag::Box) {
        const auto press = Im(panel.Press);
        const auto box_min = ImMin(press, mouse), box_max = ImMax(press, mouse);
        draw.AddRectFilled(box_min, box_max, U32(colors.BoxSelect));
        draw.AddRect(box_min, box_max, U32(colors.BoxSelectBorder));
    }
    draw.PopClipRect();
    if (view.Geometry) DrawAddMenu(panel, theme, cell, mesh_entity, *view.Geometry, interactive);
    else {
        BeginDisabled(!interactive);
        if (BeginPopup("##AddNode")) {
            DrawSceneAddMenu(r, view.Network);
            EndPopup();
        }
        if (BeginPopup("##NodeMenu")) {
            if (panel.ContextNode < nodes.size()) {
                if (panel.ContextSocket == UINT16_MAX) DrawSceneNodeMenu(r, panel, nodes[panel.ContextNode].Item);
                else if (panel.ContextSocket < NodeSockets(view, panel.ContextNode).size()) {
                    if (const auto *reference = std::get_if<SceneReference>(&BindingOf(view, panel.ContextNode, panel.ContextSocket))) DrawReferenceMenu(r, view.Network, *reference);
                }
            }
            EndPopup();
        }
        EndDisabled();
    }
    DrawPortPicker(r, panel, mesh_entity, graph, interactive);
}

bool ObjectChoice(const state::Registry &r, const char *label, state::Entity &value) {
    bool changed = false;
    const auto current = r.valid(value) ? value : state::Null;
    const auto objects = [&] {
        std::vector<state::Entity> choices{state::Null};
        for (const auto [e, instance] : r.view<const Instance>().each()) {
            if (DrawsMesh(r, instance.Entity)) choices.push_back(e);
        }
        return choices;
    };
    const auto name = [&](state::Entity e) { return e == state::Null ? std::string{"None"} : GetName(r, e); };
    ui::ChoiceCombo(label, current, objects, name, [&](state::Entity e) {
        value = e;
        changed = true;
    });
    return changed;
}

// The panel holds the text of the one active field.
void NameField(NodesPanel &panel, state::Entity node, const char *label, const AttributeName &name, bool interactive) {
    const auto id = GetID(label);
    if (panel.NameEditId != id) panel.NameEdit = name;
    InputText(label, panel.NameEdit.data(), panel.NameEdit.size());
    if (IsItemActive()) panel.NameEditId = id;
    if (!IsItemDeactivated()) return;
    panel.NameEditId = 0;
    if (IsItemDeactivatedAfterEdit() && interactive) action::Emit(action::nodes::SetAttributeName{node, std::string{panel.NameEdit.data()}});
}

template<auto M, auto... Nested>
bool ReadsLink(std::span<const Socket> linked) {
    const auto member = [] {
        if constexpr (sizeof...(Nested) == 0) return NoField;
        else return uint8_t(field::IndexOf<field::detail::First<Nested...>>);
    }();
    return std::ranges::any_of(linked, [&](Socket socket) { return socket.Carries(uint8_t(field::IndexOf<M>), member); });
}

template<typename F>
concept LeafParameter = std::same_as<F, AttributeName> || std::same_as<F, state::Entity> || std::same_as<F, mat4> || std::same_as<F, FloatMathOp>;

// A combo of the operations under the headings Blender's Math node uses.
bool OperationChoice(const char *label, FloatMathOp &value) {
    if (!BeginCombo(label, field::EnumLabel(value), ImGuiComboFlags_HeightLargest)) return false;
    bool changed = false;
    for (const auto op : field::EnumValues<FloatMathOp>) {
        if (const auto *heading = OperationHeading(op)) SeparatorText(heading);
        if (Selectable(field::EnumLabel(op), op == value)) {
            value = op;
            changed = true;
        }
    }
    EndCombo();
    return changed;
}

// Edits a node's parameter fields through Update actions at their byte offsets in the node component.
template<typename K, auto... Prefix>
struct ParameterEdit : ui::FieldWidgets<Prefix...> {
    state::Registry &R;
    NodesPanel &Panel;
    state::Entity Node;
    const K &Value;
    const std::vector<Socket> &Linked;
    bool Interactive;
    const char *Status;

    template<auto... More>
    ParameterEdit<K, Prefix..., More...> Sub() const { return {{}, R, Panel, Node, Value, Linked, Interactive, nullptr}; }

    template<auto... Ms>
    auto Read() const { return ui::ReadChain<Prefix..., Ms...>(Value); }

    template<auto... Ms>
    bool Run(auto widget, bool delta_capable = false) {
        using F = action::detail::last_field<Prefix..., Ms...>;
        const bool reads_link = ReadsLink<Prefix..., Ms...>(Linked);
        Status = reads_link ? "Linked" : nullptr;
        BeginDisabled(reads_link);
        const bool changed = ui::detail::RunField(R, Node, ChannelTarget{}, Read<Ms...>(), widget, /*selection=*/false, delta_capable && action::DeltaField<F>, [&](action::Target, const F &v) {
            action::Emit(action::Update<F>{Node, state::Key<K>(), action::detail::FieldOffset<Prefix..., Ms...>(), v}, action::Phase::Stage);
        });
        EndDisabled();
        return changed;
    }

    template<auto M>
        requires LeafParameter<field::Type<M>>
    void Leaf() {
        using F = field::Type<M>;
        const char *label = field::ChainLabel<Prefix..., M>.c_str();
        if constexpr (std::same_as<F, AttributeName>) {
            Status = nullptr;
            NameField(Panel, Node, label, ui::ReadChain<Prefix..., M>(Value), Interactive);
        } else if constexpr (std::same_as<F, mat4>) {
            Status = nullptr;
            const auto &m = ui::ReadChain<Prefix..., M>(Value);
            for (int row = 0; row < 4; ++row) {
                std::array values{m[0][row], m[1][row], m[2][row], m[3][row]};
                PushID(row);
                InputFloat4(row == 0 ? label : "##row", values.data(), "%.3f", ImGuiInputTextFlags_ReadOnly);
                PopID();
            }
        } else if constexpr (std::same_as<F, FloatMathOp>) {
            Run<M>([&](FloatMathOp &v) { return OperationChoice(label, v); });
        } else {
            Run<M>([&](state::Entity &v) { return ObjectChoice(R, label, v); });
        }
        if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ui::Hovered = {
                .Label = label,
                .Path = field::ChainPath<Prefix..., M>.c_str(),
                .Type = std::same_as<F, AttributeName> ? "string" : field::TypeName<F>.c_str(),
                .Help = field::ChainSpecOf<Prefix..., M>.Help,
                .Status = Status,
            };
        }
    }
};

state::Entity ViewportObject(const state::Registry &r, state::Entity mesh) {
    const auto object = r.Resources.get<const PanelsState>().Nodes.Canvas.Object;
    if (const auto *instance = r.try_get<const Instance>(object); instance && instance->Entity == mesh) return object;
    for (const auto [e, instance] : r.view<const Instance>().each()) {
        if (instance.Entity == mesh && InScene(r, e, FrameScene(r)) && !r.all_of<Hidden>(e)) return e;
    }
    return state::Null;
}

void DrawInlineSelection(const state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, uint32_t node, bool interactive) {
    if (plan.Nodes[node].Kind == KindOf<Subnet>) return;
    const auto sockets = NodeSockets(plan, node);
    const auto socket = std::ranges::find(sockets, SocketType::Selection, &Socket::Type);
    if (socket == sockets.end()) return;
    SeparatorText("Selection");
    if (IsLinked(plan, node, uint16_t(socket - sockets.begin()))) {
        TextDisabled("Linked");
        return;
    }
    const auto entity = plan.Nodes[node].Entity;
    const auto *selection = r.try_get<const InlineSelection>(entity);
    const bool every = !selection || selection->All;
    const auto counts = every ? std::array<uint32_t, 3>{} : std::array{SelectedCount(selection->Vertices), SelectedCount(selection->Edges), SelectedCount(selection->Faces)};
    if (every) TextUnformatted("All elements");
    else if (selection->Published) TextUnformatted("Published by the input");
    else if (counts == std::array<uint32_t, 3>{}) TextUnformatted("No elements");
    else Text("%u vertices, %u edges, %u faces", counts[0], counts[1], counts[2]);
    const auto *task = r.try_get<const SelectionEditing>(state::Document);
    if (task && task->Node == entity) {
        if (Button("Done") && interactive) action::Emit(action::nodes::FinishSelection{});
        SameLine();
        if (Button("Cancel") && interactive) action::Cancel();
        return;
    }
    const auto input = ResolveSocket(r, InputOf(r, entity, 0u), true).Source;
    const auto object = ViewportObject(r, mesh_entity);
    BeginDisabled(input.FromNode == state::Null || object == state::Null || task);
    if (Button("Edit selection...") && interactive) action::Emit(action::nodes::EditSelection{entity, object});
    EndDisabled();
    ui::Describe("Edit selection...", "Select input vertices, edges or faces over the displayed result. Enter commits one undo step; Esc cancels.");
    SameLine();
    BeginDisabled(every);
    if (Button("All") && interactive) action::Emit(action::nodes::SetInlineSelection{entity, std::make_unique<InlineSelection>(InlineSelection{.Domain = selection->Domain, .All = true})});
    EndDisabled();
    ui::Describe("All", "Set the node's selection to all elements");
}

void DrawParameters(state::Registry &r, NodesPanel &panel, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    const auto node = ActiveIndex(r, plan);
    if (node == NoNode) {
        TextWrapped("Right click or press Shift+A to add a node.");
        return;
    }
    const auto [active, kind] = plan.Nodes[node];
    PushID(int(state::Integral(active)));
    SeparatorText(OperationTitle(r, active, kind));
    if (ResultSocket(r, active)) {
        ui::ActionField("Display", r.get<const GeometryProcedure>(mesh_entity).DisplayNode == active, [&](bool displayed) { return action::nodes::SetDisplay{mesh_entity, displayed ? active : state::Null}; }, {.Help = "Display this node's result. Turn off to display no result."}, interactive);
    }
    if (BypassInput(kind) || kind == KindOf<Subnet>) {
        if (ResultSocket(r, active)) SameLine();
        ui::ActionField("Bypass", r.all_of<NodeBypassed>(active), [&](bool bypassed) { return action::nodes::SetBypass{active, bypassed}; }, {.Help = "Pass the compatible input through without applying this node."}, interactive);
    }
    if (kind == KindOf<Subnet>) DrawSubnetParameters(r, panel, active, interactive);
    const auto parent = ContainerOf(r, active);
    if (parent != state::Null && BeginMenu("Expose socket")) {
        const auto sockets = NodeSockets(plan, node);
        for (uint16_t s = 0; s < sockets.size(); ++s) {
            PushID(s);
            if (MenuItem(LabelOf(sockets[s])) && interactive) action::Emit(action::nodes::ExposeSocket{parent, active, SocketId(plan, node, s), {}});
            PopID();
        }
        EndMenu();
    }
    const auto linked = LinkedParameters(plan, node);
    if (std::ranges::any_of(HandleSpecs, [&](const auto &spec) { return spec.Kind == kind; })) {
        const auto object = ViewportObject(r, mesh_entity);
        BeginDisabled(object == state::Null);
        if (BeginMenu("Viewport handle")) {
            for (uint8_t i = 0; i < HandleSpecs.size(); ++i) {
                const auto &spec = HandleSpecs[i];
                if (spec.Kind != kind) continue;
                const bool wired = std::ranges::any_of(linked, [&](Socket socket) { return spec.Reads(socket); });
                if (MenuItem(spec.Label, nullptr, false, !wired) && interactive) action::Emit(action::nodes::SetOperationHandle{active, object, HandleParameter(i)});
            }
            EndMenu();
        }
        EndDisabled();
        if (const auto *handle = r.try_get<const OperationHandle>(state::Document); handle && handle->Node == active) {
            SameLine();
            if (Button("Stop handle") && interactive) action::Emit(action::nodes::SetOperationHandle{});
        }
    }
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K>) ui::DrawFields<K>(ParameterEdit<K>{{}, r, panel, active, r.get<const K>(active), linked, interactive, nullptr});
    });
    DrawInlineSelection(r, mesh_entity, plan, node, interactive);
    if (const auto *error = r.try_get<const NodeError>(active)) {
        const auto color = r.get<const NodeTheme>(state::Document).Colors.Error;
        PushStyleColor(ImGuiCol_Text, {color.x, color.y, color.z, color.w});
        TextWrapped("%s", FaultTexts[uint8_t(error->Fault)]);
        PopStyleColor();
    }
    PopID();
}
const CookPlan *NetworkPlan(const state::Registry &r, NodesPanel &panel, state::Entity &owner) {
    auto network = EditorNetwork(r, panel);
    while (const auto *context = r.try_get<const HierarchyGeometry>(network)) network = context->Source;
    if (r.all_of<HierarchyGeometry>(panel.Canvas.Network)) panel.Canvas.Network = network;
    owner = r.all_of<Subnet>(network) ? r.get<const ProcedureNode>(network).Procedure : network;
    const auto *compiled = r.try_get<const CookPlan>(owner);
    if (!compiled) return nullptr;
    if (!panel.NetworkView || panel.Epoch != r.Epoch || panel.ViewedNetwork != network || panel.NetworkView->Revision != compiled->Revision) {
        panel.NetworkView = std::make_shared<CookPlan>(ProjectNetwork(r, owner, network == owner ? state::Null : network));
        panel.NetworkView->Revision = compiled->Revision;
        panel.ViewedNetwork = network;
    }
    return panel.NetworkView.get();
}
} // namespace

void DrawNodesPanel(state::Registry &r, NodesPanel &panel, bool interactive) {
    if (panel.Epoch != r.Epoch) {
        panel.Epoch = r.Epoch;
        panel.NetworkView.reset();
        panel.Drag = NodeDrag::None;
        panel.PendingLinks.clear();
        panel.LinkAfter = NoNode;
    }
    state::Entity mesh_entity;
    const auto *plan = NetworkPlan(r, panel, mesh_entity);
    if (interactive && panel.LinkAfter != NoNode) {
        if (plan && plan->Nodes.size() > panel.LinkAfter && panel.DragNode < plan->Nodes.size()) {
            action::Emit(LinkOf(*plan, mesh_entity, panel.DragNode, panel.DragSocket, panel.LinkAfter, panel.LinkSocket), action::Phase::Stage);
        }
        action::Commit();
        panel.LinkAfter = NoNode;
    } else if (interactive && !panel.PendingLinks.empty()) {
        action::Emit(panel.PendingLinks.front(), action::Phase::Stage);
        panel.PendingLinks.erase(panel.PendingLinks.begin());
        if (panel.PendingLinks.empty()) action::Commit();
    }
    panel.Hovered = false;
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        DrawNetworkNavigation(r, panel);
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
        BeginChild("Canvas", {0.f, 0.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
        PopStyleVar();
        const auto network = EditorNetwork(r, panel);
        const auto *current = NetworkPlan(r, panel, mesh_entity);
        const auto view = current ? GeometryNodeDefinitions(r, *current, network) : SceneNodeDefinitions(r, network);
        DrawCanvas(r, r.Resources.get<FrameState>(), panel, r.get<const NodeTheme>(state::Document), mesh_entity, view, interactive);
        EndChild();
    }
    End();
}

void DrawParametersPanel(state::Registry &r, NodesPanel &nodes, Panel &panel, bool interactive) {
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        state::Entity mesh_entity;
        const auto *plan = NetworkPlan(r, nodes, mesh_entity);
        if (!plan) DrawNetworkParameters(r, nodes);
        else DrawParameters(r, nodes, mesh_entity, *plan, interactive);
    }
    End();
}
void DrawEditor(auto &e, std::type_identity<CellSize>) {
    BeginDisabled(e.template Read<&CellSize::FollowFrameHeight>());
    e.template Draw<&CellSize::Points>();
    EndDisabled();
    SameLine();
    e.template Draw<&CellSize::FollowFrameHeight>();
}
void DrawNodeStyle(state::Registry &r) {
    if (Button("Reset")) action::Emit(action::view::ResetNodeTheme{});
    ui::DrawFields<NodeTheme>(ui::Edit{r, state::Document});
}
} // namespace procedure
