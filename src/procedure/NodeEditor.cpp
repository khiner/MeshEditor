#include "procedure/NodeEditor.h"

#include "action/View.h"
#include "mesh/Mesh.h"
#include "procedure/NodeTheme.h"
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
constexpr float LinkReach{4.f}; // A link takes the mouse within this many pixels of its wire.
constexpr float WireStraightening{3.5f}; // A link's handles straighten as the link nears horizontal, as in Blender.
constexpr float MinZoom{.25f}, MaxZoom{3.f};
constexpr float TrackpadPoints{10.f}; // The platform reports a trackpad's scroll at a tenth of its points.
constexpr float ZoomStep{1.1f}, TrackpadStepPoints{100.f}; // A zoom step is one mouse wheel notch, or 100 points of trackpad scroll.

enum class KindGroup : uint8_t { Sources,
                                 Operators,
                                 Combine,
                                 Attributes,
                                 Values,
                                 Generators,
                                 Instances };
constexpr const char *GroupNames[]{"Sources", "Operators", "Combine", "Attributes", "Values", "Generators", "Instances"};
using C = NodeThemeColors;
constexpr vec4 C::*GroupColors[]{&C::SourcesTitle, &C::OperatorsTitle, &C::CombineTitle, &C::AttributesTitle, &C::ValuesTitle, &C::GeneratorsTitle, &C::InstancesTitle};
using AttributeKinds = TypeList<Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, StoreSelection, SetPosition>;
using InstanceKinds = TypeList<MeshToPoints, InstanceOnPoints, RealizeInstances>;
template<typename K> constexpr KindGroup GroupOf() {
    if constexpr (std::same_as<K, Source> || std::same_as<K, ObjectInfo>) return KindGroup::Sources;
    else if constexpr (std::same_as<K, Join> || std::same_as<K, Boolean>) return KindGroup::Combine;
    else if constexpr (Generator<K>) return KindGroup::Generators;
    else if constexpr (InTypeList<K, AttributeKinds>) return KindGroup::Attributes;
    else if constexpr (ValueKind<K>) return KindGroup::Values;
    else if constexpr (InTypeList<K, InstanceKinds>) return KindGroup::Instances;
    else return KindGroup::Operators;
}
constexpr auto KindGroups{[]<typename... Ks>(TypeList<Ks...>) { return std::array{GroupOf<Ks>()...}; }(NodeKinds{})};
constexpr auto KindTitles{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Title<Ks>.c_str()...}; }(NodeKinds{})};

template<typename K> constexpr const char *Help{};
template<> constexpr const char *Help<Source>{"Emits the mesh's source geometry"};
template<> constexpr const char *Help<Join>{"Appends the second input's geometry to the first"};
template<> constexpr const char *Help<Boolean>{"Combines two closed meshes by union, difference, or intersection"};
template<> constexpr const char *Help<primitive::Plane>{"Generates a rectangle in the XZ plane, facing +Y"};
template<> constexpr const char *Help<primitive::Circle>{"Generates a disc of triangles in the XZ plane, facing +Y"};
template<> constexpr const char *Help<primitive::Cube>{"Generates a box centered on the origin"};
template<> constexpr const char *Help<primitive::IcoSphere>{"Generates a sphere by subdividing an icosahedron"};
template<> constexpr const char *Help<primitive::UVSphere>{"Generates a sphere of slices around the Y axis and stacks along it"};
template<> constexpr const char *Help<primitive::Torus>{"Generates a torus around the Y axis"};
template<> constexpr const char *Help<primitive::Cylinder>{"Generates a capped cylinder along the Y axis, centered on the origin"};
template<> constexpr const char *Help<primitive::Cone>{"Generates a capped cone along the Y axis, centered on the origin"};
template<> constexpr const char *Help<action::mesh::Delete>{"Deletes the selected vertices, edges, or faces"};
template<> constexpr const char *Help<action::mesh::Merge>{"Merges the selected vertices into one, or merges nearby pairs"};
template<> constexpr const char *Help<action::mesh::Extrude>{"Extrudes the selected elements in place, joined to the geometry they came from"};
template<> constexpr const char *Help<action::mesh::Duplicate>{"Duplicates the selected faces in place onto copied vertices"};
template<> constexpr const char *Help<action::mesh::Split>{"Disconnects the selected faces from the unselected geometry around them"};
template<> constexpr const char *Help<action::mesh::Dissolve>{"Removes the selected elements and joins the faces around them"};
template<> constexpr const char *Help<action::mesh::Subdivide>{"Cuts the selected edges and splits their faces into smaller faces"};
template<> constexpr const char *Help<action::mesh::Triangulate>{"Splits the selected faces into triangles"};
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
template<> constexpr const char *Help<Spin>{"Extrudes the selection in steps around an axis"};
template<> constexpr const char *Help<action::mesh::ExtrudeRepeat>{"Extrudes the selection repeatedly along an offset"};
template<> constexpr const char *Help<Bisect>{"Cuts the mesh along a plane and deletes the faces on the cleared sides"};
template<> constexpr const char *Help<action::mesh::Symmetrize>{"Mirrors one side of the mesh onto the other across an axis plane through the mesh origin"};
template<> constexpr const char *Help<action::mesh::FillHoles>{"Fills boundary loops with faces"};
template<> constexpr const char *Help<action::mesh::ConvexHull>{"Builds the convex hull of the selected vertices"};
template<> constexpr const char *Help<action::mesh::Decimate>{"Reduces the selected region's triangle count by collapsing edges"};
template<> constexpr const char *Help<action::mesh::RecalculateNormals>{"Makes the selected faces' normals point consistently outside or inside"};
template<> constexpr const char *Help<action::mesh::Rip>{"Splits the faces along the selected edges"};
template<> constexpr const char *Help<action::mesh::SplitNonplanarFaces>{"Splits the selected faces that bend more than Angle along their diagonals"};
template<> constexpr const char *Help<action::mesh::SplitConcaveFaces>{"Splits the selected concave faces into convex faces"};
template<> constexpr const char *Help<action::mesh::ConnectVertices>{"Splits faces along chords between their consecutive selected vertices"};
template<> constexpr const char *Help<action::mesh::Fill>{"Creates an edge from two selected vertices, fills the selected loops, or forms a face from the selected points"};
template<> constexpr const char *Help<action::mesh::BridgeEdgeLoops>{"Bridges two selected edge loops or chains with a strip of faces"};
template<> constexpr const char *Help<action::mesh::GridFill>{"Fills one selected closed loop with a grid of quads"};
template<> constexpr const char *Help<action::mesh::EdgeRotate>{"Rotates each selected edge between its two faces"};
template<> constexpr const char *Help<action::mesh::BeautifyFaces>{"Rearranges the selected triangles into more evenly shaped ones"};
template<> constexpr const char *Help<action::mesh::Unsubdivide>{"Reverses subdivision by removing alternate selected vertices of regular regions"};
template<> constexpr const char *Help<action::mesh::LoopCut>{"Cuts the edge ring through the active selected edge, or the first selected edge"};
template<> constexpr const char *Help<Knife>{"Cuts every edge whose projection in a view crosses a screen segment"};
template<> constexpr const char *Help<action::mesh::RotateUVs>{"Rotates the UVs of the selected faces around their corners"};
template<> constexpr const char *Help<action::mesh::ReverseUVs>{"Reverses the order of the UVs around the selected faces"};
template<> constexpr const char *Help<action::mesh::RotateColors>{"Rotates the colors of the selected faces around their corners"};
template<> constexpr const char *Help<action::mesh::ReverseColors>{"Reverses the order of the colors around the selected faces"};
template<> constexpr const char *Help<MarkSharp>{"Marks the selected edges sharp or smooth, or shades the selected faces flat or smooth"};
template<> constexpr const char *Help<ShadeSmooth>{"Shades every face smooth and clears every sharp mark, or shades every face flat"};
template<> constexpr const char *Help<ShadeSmoothByAngle>{"Shades every face smooth and marks the edges sharp where adjacent faces meet at more than Angle"};
template<> constexpr const char *Help<TransformElements>{"Moves, rotates, and scales the selected vertices around a pivot"};
template<> constexpr const char *Help<Position>{"Emits each vertex's position"};
template<> constexpr const char *Help<Normal>{"Emits each vertex's or face's normal"};
template<> constexpr const char *Help<Index>{"Emits each element's index in its domain, counting live elements only"};
template<> constexpr const char *Help<NamedAttribute>{"Emits the values of the attribute with the given name and type"};
template<> constexpr const char *Help<VectorMath>{"Applies a vector operation to A and B per element"};
template<> constexpr const char *Help<FloatMath>{"Applies a float operation to A and B per element"};
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
template<> constexpr const char *Help<TransformGeometry>{"Moves, rotates, and scales the whole geometry about the mesh origin"};
template<> constexpr const char *Help<RotationValue>{"Emits a constant rotation"};
constexpr auto KindHelp{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Help<Ks>...}; }(NodeKinds{})};
static_assert(std::ranges::none_of(KindHelp, [](const char *help) { return help == nullptr; }), "Every node kind needs a Help sentence.");

constexpr const char *FaultTexts[]{"The selection does not fit this operation.", "A parameter is invalid.", "An attribute input is missing or has the wrong type or domain.", "An input mesh is not a closed surface.", "The boolean produced no geometry.", "The node reads no mesh: the Object Info object has none, or the Instance input is unlinked."};

constexpr const char *SocketLabels[]{"Geometry", "Selection", "Attribute", "Instances", "Value"};
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
// Draw per-element value sockets as diamonds and uniform sockets as circles.
// Use the hover color for the outline while hovered.
void DrawSocket(ImDrawList &draw, const NodeTheme &theme, ImVec2 at, float radius, Socket socket, vec4 color, bool hovered) {
    const auto fill = U32(color), outline = U32(hovered ? theme.Colors.Hovered : theme.Colors.Outline);
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
// Its parameters sit in a tray under the fixed inputs, one chevron row when closed and one row per parameter when open.
// The row an input socket sits on, and the rows the inputs take.
uint32_t InputRow(uint16_t kind, uint16_t socket, bool expanded) { return expanded ? socket : std::min<uint32_t>(socket, ParameterRange(kind).first); }
uint32_t InputRows(uint16_t kind, bool expanded) { return expanded ? InputCount[kind] : std::min<uint32_t>(InputCount[kind], ParameterRange(kind).first + 1u); }

// A grid cell's side in canvas points.
float CellOf(const NodeTheme &theme) { return theme.Sizes.Cell.FollowFrameHeight ? GetFrameHeight() : theme.Sizes.Cell.Points; }
uint32_t CellsOf(float cell, float points) { return std::max(1u, uint32_t(std::ceil(points / cell))); }

// A node's cells: its top left and size, and whether it has inputs and outputs, which keep a free cell beside them for wires.
struct Box {
    uint32_t X, Y;
    uint16_t W, H;
    bool In, Out;
};
// A node of the kind takes a title row over max(input rows, output rows), as wide as its title and its widest row of labels.
// A socket label takes at most MaxLabelCells and draws with an ellipsis past them.
Box BoxOf(const NodeTheme &theme, float cell, uint16_t kind, bool expanded) {
    const auto sockets = KindSockets(kind);
    const auto [first, end] = ParameterRange(kind);
    const uint32_t inputs = InputCount[kind], outputs = uint32_t(sockets.size()) - inputs;
    const auto label_cells = [&](uint16_t s) {
        const float indent = s >= first && s < end ? theme.Sizes.ParameterIndent : 0.f;
        return std::min(CellsOf(cell, theme.Sizes.SocketRadius + theme.Sizes.LabelSpacing * 2.f + indent + CalcTextSize(LabelOf(sockets[s])).x), uint32_t(theme.Sizes.MaxLabelCells));
    };
    uint32_t width = CellsOf(cell, CalcTextSize(KindTitles[kind]).x + theme.Sizes.TitlePadding * 2.f);
    for (uint32_t row = 0u; row < std::max(inputs, outputs); ++row) {
        width = std::max(width, (row < inputs ? label_cells(uint16_t(row)) : 0u) + (row < outputs ? label_cells(uint16_t(inputs + row)) : 0u));
    }
    return {0, 0, uint16_t(width), uint16_t(1u + std::max({InputRows(kind, expanded), outputs, 1u})), inputs > 0u, outputs > 0u};
}
// Whether the boxes leave a free cell beside every socket column between them: right of a when a has outputs or b inputs, and the reverse.
bool Clear(const Box &a, const Box &b) {
    if (a.Y >= b.Y + b.H || b.Y >= a.Y + a.H) return true;
    return b.X >= a.X + a.W + uint32_t(a.Out || b.In) || a.X >= b.X + b.W + uint32_t(b.Out || a.In);
}
// A socket's cell column and row within a node of the kind and width.
ImVec2 SocketCell(uint16_t kind, uint16_t socket, uint16_t width, bool expanded) {
    const bool input = socket < InputCount[kind];
    return {input ? 0.f : float(width), 1.f + float(input ? InputRow(kind, socket, expanded) : socket - InputCount[kind])};
}

// The screen layout of a procedure's nodes, with cell c at Origin + c * Step.
struct Graph {
    const CookPlan &Plan;
    const NodeTheme &Theme;
    ImVec2 Origin;
    float Step; // A cell on the screen
    float Zoom;
    uint32_t Expanded;
    std::vector<Box> Boxes;
};
Graph GraphOf(const state::Registry &r, const CookPlan &plan, const NodeTheme &theme, ImVec2 origin, float cell, float zoom, uint32_t expanded) {
    // A node index a deletion left stale opens no tray.
    const bool open = expanded < plan.Nodes.size() && ParameterRange(plan.Nodes[expanded].Kind).second > ParameterRange(plan.Nodes[expanded].Kind).first;
    Graph graph{plan, theme, origin, cell * zoom, zoom, open ? expanded : NoNode, {}};
    graph.Boxes.reserve(plan.Nodes.size());
    for (uint32_t n = 0u; n < plan.Nodes.size(); ++n) {
        const auto [entity, kind] = plan.Nodes[n];
        auto box = BoxOf(theme, cell, kind, n == graph.Expanded);
        const auto position = r.get<const NodeLayout>(entity).Position;
        box.X = position.x;
        box.Y = position.y;
        graph.Boxes.push_back(box);
    }
    return graph;
}
ImRect RectOf(const Graph &graph, const Box &box) {
    return {graph.Origin + ImVec2{float(box.X), float(box.Y)} * graph.Step, graph.Origin + ImVec2{float(box.X + box.W), float(box.Y + box.H)} * graph.Step};
}
// A collapsed node's parameters chevron sits at its first parameter socket.
ImVec2 SocketAt(const Graph &graph, uint32_t node, uint16_t socket) {
    const auto &box = graph.Boxes[node];
    return graph.Origin + (ImVec2{float(box.X), float(box.Y) + .5f} + SocketCell(graph.Plan.Nodes[node].Kind, socket, box.W, node == graph.Expanded)) * graph.Step;
}
ImRect TrayRect(const Graph &graph, uint32_t node) {
    const auto [first, end] = ParameterRange(graph.Plan.Nodes[node].Kind);
    const float reach = graph.Theme.Sizes.SocketRadius * 1.2f * graph.Zoom + 3.f, row = graph.Step;
    const auto at = SocketAt(graph, node, first);
    return {{at.x - reach, at.y - row * .5f}, {at.x + reach, at.y + (float(end - first) - .5f) * row}};
}
// Whether moving the selected nodes by the offset keeps each inside the canvas, with a cell for wires left of its inputs, and clear of every other node.
bool Clears(const Graph &graph, std::span<const uint32_t> selected, int64_t dx, int64_t dy) {
    for (const auto m : selected) {
        auto box = graph.Boxes[m];
        const int64_t x = int64_t(box.X) + dx, y = int64_t(box.Y) + dy;
        if (x < int64_t(box.In) || y < 0) return false;
        box.X = uint32_t(x);
        box.Y = uint32_t(y);
        for (uint32_t n = 0u; n < graph.Boxes.size(); ++n) {
            if (!std::ranges::binary_search(selected, n) && !Clear(box, graph.Boxes[n])) return false;
        }
    }
    return true;
}
constexpr int64_t PlaceReach{64}; // Rings of cells a placement searches around the wanted offset.
// The offset nearest the wanted one that Clears, searching outward one ring of cells at a time, or none within reach.
std::optional<std::pair<int64_t, int64_t>> Place(const Graph &graph, std::span<const uint32_t> selected, int64_t dx, int64_t dy) {
    for (int64_t ring = 0; ring <= PlaceReach; ++ring) {
        for (int64_t i = -ring; i <= ring; ++i) {
            for (int64_t j = -ring; j <= ring; ++j) {
                if ((std::abs(i) == ring || std::abs(j) == ring) && Clears(graph, selected, dx + i, dy + j)) return std::pair{dx + i, dy + j};
            }
        }
    }
    return {};
}
bool IsLinked(const CookPlan &plan, uint32_t node, uint16_t socket) { return plan.Inputs[plan.InputFirst[node] + socket].FromNode != NoNode; }
std::vector<Socket> LinkedParameters(const CookPlan &plan, uint32_t node) {
    const auto kind = plan.Nodes[node].Kind;
    const auto [first, end] = ParameterRange(kind);
    std::vector<Socket> linked;
    for (auto s = first; s < end; ++s) {
        if (IsLinked(plan, node, s)) linked.push_back(KindSockets(kind)[s]);
    }
    return linked;
}

using Wire = std::array<ImVec2, 4>;
Wire WireOf(ImVec2 from, ImVec2 to, float curving) {
    const float dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const float straightening = dx > 0.f ? std::min(1.f, dy / dx * WireStraightening) : 1.f;
    const ImVec2 handle{curving * dx * straightening, 0.f};
    return {from, from + handle, to - handle, to};
}
ImVec2 NearestOnWire(const Wire &w, ImVec2 p) { return ImBezierCubicClosestPoint(w[0], w[1], w[2], w[3], p, 24); }
// Visits each linked input as `f(wire, link, node, input)`.
void ForEachWire(const Graph &graph, auto &&f) {
    const auto &plan = graph.Plan;
    for (uint32_t n = 0u, slot = 0u; n < plan.Nodes.size(); ++n) {
        for (uint16_t i = 0u; i < InputCount[plan.Nodes[n].Kind]; ++i, ++slot) {
            if (const auto link = plan.Inputs[slot]; link.FromNode != NoNode) f(WireOf(SocketAt(graph, link.FromNode, link.FromSocket), SocketAt(graph, n, i), graph.Theme.Sizes.WireCurving), link, n, i);
        }
    }
}

enum class HitType : uint8_t { None,
                               Socket,
                               Parameters, // A node's parameters tray header
                               Node,
                               Link };
struct Hit {
    HitType Type{HitType::None};
    uint32_t Node{NoNode};
    uint16_t Socket{};
};
// The expanded node comes first, then sockets before node bodies, the last drawn node first, and links come last.
Hit HitTest(const Graph &graph, ImVec2 mouse) {
    const float reach = std::max(graph.Theme.Sizes.SocketRadius * graph.Zoom, 4.f) + 3.f;
    const auto sockets = [&](uint32_t n) -> std::optional<Hit> {
        const auto kind = graph.Plan.Nodes[n].Kind;
        const auto [first, end] = ParameterRange(kind);
        const bool collapsed = n != graph.Expanded;
        const auto near = [&](uint16_t s) { return ImLengthSqr(SocketAt(graph, n, s) - mouse) <= reach * reach; };
        for (uint16_t s = 0u; s < SocketCount[kind]; ++s) {
            if (!(collapsed && s >= first && s < end) && near(s)) return Hit{HitType::Socket, n, s};
        }
        if (first < end && collapsed && near(first)) return Hit{HitType::Parameters, n, first};
        return {};
    };
    if (graph.Expanded != NoNode) {
        if (const auto hit = sockets(graph.Expanded)) return *hit;
        if (RectOf(graph, graph.Boxes[graph.Expanded]).Contains(mouse)) return {HitType::Node, graph.Expanded, 0u};
    }
    for (auto n = uint32_t(graph.Boxes.size()); n-- > 0u;) {
        if (const auto hit = sockets(n)) return *hit;
    }
    for (auto n = uint32_t(graph.Boxes.size()); n-- > 0u;) {
        if (RectOf(graph, graph.Boxes[n]).Contains(mouse)) return {HitType::Node, n, 0u};
    }
    Hit hit;
    ForEachWire(graph, [&](const Wire &wire, PlanLink, uint32_t node, uint16_t input) {
        if (hit.Type == HitType::None && ImLengthSqr(NearestOnWire(wire, mouse) - mouse) <= LinkReach * LinkReach) hit = {HitType::Link, node, input};
    });
    return hit;
}

bool Joins(uint16_t held_kind, uint16_t held_socket, uint16_t kind, uint16_t socket) {
    const auto held = KindSockets(held_kind)[held_socket], other = KindSockets(kind)[socket];
    if (held.Direction == other.Direction) return false;
    return held.Direction == SocketDirection::Out ? Links(held, other) : Links(other, held);
}
bool Accepts(const CookPlan &plan, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    return node != held_node && Joins(plan.Nodes[held_node].Kind, held_socket, plan.Nodes[node].Kind, socket);
}
std::vector<uint16_t> LinkingSockets(const CookPlan &plan, uint32_t held_node, uint16_t held_socket, uint32_t node, std::pair<uint16_t, uint16_t> range) {
    std::vector<uint16_t> sockets;
    for (auto s = range.first; s < range.second; ++s) {
        if (Accepts(plan, held_node, held_socket, node, s)) sockets.push_back(s);
    }
    return sockets;
}
action::nodes::Link LinkOf(const CookPlan &plan, state::Entity mesh_entity, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    const bool from_held = KindSockets(plan.Nodes[held_node].Kind)[held_socket].Direction == SocketDirection::Out;
    const auto held = plan.Nodes[held_node].Entity, other = plan.Nodes[node].Entity;
    return from_held ? action::nodes::Link{mesh_entity, held, held_socket, other, socket} : action::nodes::Link{mesh_entity, other, socket, held, held_socket};
}
// Drops the held wire on the node's sockets in the range: it links with the one socket that takes it, or the port picker offers the sockets that do.
void DropOnNode(NodesPanel &panel, const CookPlan &plan, state::Entity mesh_entity, uint32_t node, std::pair<uint16_t, uint16_t> range) {
    const auto sockets = LinkingSockets(plan, panel.DragNode, panel.DragSocket, node, range);
    if (sockets.size() == 1u) {
        action::Emit(LinkOf(plan, mesh_entity, panel.DragNode, panel.DragSocket, node, sockets.front()));
    } else if (sockets.size() > 1u) {
        panel.PickNode = node;
        panel.PickRange = range;
        OpenPopup("##PickPort");
    }
}

// Begin moving selected nodes after the node count exceeds `after`.
void Grab(NodesPanel &panel, uint32_t after) {
    panel.Drag = NodeDrag::Grab;
    panel.DragNode = NoNode;
    panel.GrabAfter = after;
    panel.Press = vec2{GetIO().MousePos};
    panel.Moved = false;
}

// The entities of the plan's nodes at the indices.
std::vector<state::Entity> EntitiesOf(const CookPlan &plan, std::span<const uint32_t> indices) {
    std::vector<state::Entity> entities;
    entities.reserve(indices.size());
    for (const auto n : indices) entities.push_back(plan.Nodes[n].Entity);
    return entities;
}
std::vector<uint32_t> SelectedNodes(const state::Registry &r, const CookPlan &plan) {
    std::vector<uint32_t> selected;
    for (uint32_t n = 0u; n < plan.Nodes.size(); ++n) {
        if (r.all_of<NodeSelected>(plan.Nodes[n].Entity)) selected.push_back(n);
    }
    return selected;
}

// Connect adjacent selected nodes in horizontal order, using the first compatible input.
// Skip cycles and occupied inputs unless `replace` is set.
std::vector<action::nodes::Link> MakeLinks(const state::Registry &r, state::Entity mesh_entity, CookPlan plan, std::vector<uint32_t> nodes, bool replace) {
    std::ranges::stable_sort(nodes, {}, [&](uint32_t n) { return r.get<const NodeLayout>(plan.Nodes[n].Entity).Position.x; });
    const auto &input_first = plan.InputFirst;
    std::vector<action::nodes::Link> links;
    for (size_t i = 0u; i + 1u < nodes.size(); ++i) {
        auto from = nodes[i], to = nodes[i + 1u];
        if (InputCount[plan.Nodes[to].Kind] == 0u || SocketCount[plan.Nodes[from].Kind] == InputCount[plan.Nodes[from].Kind]) std::swap(from, to);
        if (Upstream(plan, from, [](uint32_t) { return true; })[to]) continue;
        const auto from_kind = plan.Nodes[from].Kind, to_kind = plan.Nodes[to].Kind;
        const auto first_link = [&](bool linked) -> std::optional<action::nodes::Link> {
            for (uint16_t input = 0u; input < InputCount[to_kind]; ++input) {
                const auto current = plan.Inputs[input_first[to] + input];
                if ((current.FromNode != NoNode) != linked) continue;
                for (uint16_t output = InputCount[from_kind]; output < SocketCount[from_kind]; ++output) {
                    if (Links(KindSockets(from_kind)[output], KindSockets(to_kind)[input]) && (current.FromNode != from || current.FromSocket != output)) return action::nodes::Link{mesh_entity, plan.Nodes[from].Entity, output, plan.Nodes[to].Entity, input};
                }
            }
            return {};
        };
        auto link = first_link(false);
        if (!link && replace) link = first_link(true);
        if (!link) continue;
        plan.Inputs[input_first[to] + link->ToSocket] = {from, link->FromSocket};
        links.push_back(*link);
    }
    return links;
}

// Record the insertion input, the wire point nearest the cursor, and the cells the reader moves right to clear the node.
struct Insert {
    uint32_t ToNode;
    uint16_t ToSocket;
    ImVec2 At;
    uint32_t Shift;
};
// Return the nearest link the node inserts into.
std::optional<Insert> InsertTarget(const Graph &graph, uint32_t node, ImVec2 mouse) {
    std::optional<Insert> insert;
    float nearest = LinkReach * LinkReach;
    ForEachWire(graph, [&](const Wire &wire, PlanLink, uint32_t to_node, uint16_t input) {
        const auto at = NearestOnWire(wire, mouse);
        if (const float distance = ImLengthSqr(at - mouse); distance <= nearest && Inserts(graph.Plan, node, to_node, input)) {
            nearest = distance;
            const auto &box = graph.Boxes[node], &reader = graph.Boxes[to_node];
            const uint32_t clear = box.X + box.W + 1u;
            insert = Insert{to_node, input, at, clear > reader.X ? clear - reader.X : 0u};
        }
    });
    return insert;
}

// Emit canvas actions and insert a dragged node when released over an insertion target.
void Interact(const state::Registry &r, NodesPanel &panel, state::Entity mesh_entity, const Graph &graph, const std::vector<uint32_t> &selected, const Hit &hit, const std::optional<Insert> &insert) {
    const auto &io = GetIO();
    const auto &plan = graph.Plan;
    const auto &nodes = plan.Nodes;
    const auto mouse = io.MousePos;
    const bool mouse_moved = io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f;
    const auto select = [&](const std::vector<uint32_t> &selection, uint32_t active, bool display = false) { action::Emit(action::nodes::SelectNodes{mesh_entity, EntitiesOf(plan, selection), active == NoNode ? state::Null : nodes[active].Entity, display}); };
    const auto open_add = [&](bool linking) {
        const auto cell = (mouse - graph.Origin) / graph.Step;
        panel.AddAt = {std::max(std::floor(cell.x), 0.f), std::max(std::floor(cell.y), 0.f)};
        panel.Linking = linking;
        OpenPopup("##AddNode");
    };
    // Snap the held nodes to the mouse's cell offset from the drag start, or to the nearest clear offset, or hold.
    const auto follow = [&](bool first) {
        const auto &held = graph.Boxes[panel.DragNode];
        const int64_t wanted_x = int64_t(panel.MoveStart.x) + std::lround((mouse.x - panel.Press.x) / graph.Step), wanted_y = int64_t(panel.MoveStart.y) + std::lround((mouse.y - panel.Press.y) / graph.Step);
        const auto placed = Place(graph, selected, wanted_x - int64_t(held.X), wanted_y - int64_t(held.Y));
        if (!placed) return;
        const uvec2 at{uint32_t(int64_t(held.X) + placed->first), uint32_t(int64_t(held.Y) + placed->second)};
        if (first || at != uvec2{held.X, held.Y}) action::Emit(action::UpdateOf<&NodeLayout::Position>(action::OnSelectedDelta{}, at), action::Phase::Stage);
    };
    // Commit staged move or grab actions together with insertion, or cancel the whole operation.
    const auto end_move = [&](bool commit) {
        if (panel.Moved && commit) {
            if (insert) action::Emit(action::nodes::InsertNode{mesh_entity, nodes[panel.DragNode].Entity, nodes[insert->ToNode].Entity, insert->ToSocket, insert->Shift}, action::Phase::Stage);
            action::Commit();
        } else if (panel.Moved) {
            action::Cancel();
        }
        panel.Drag = NodeDrag::None;
    };
    if (panel.Drag == NodeDrag::Grab) {
        if (panel.DragNode == NoNode) {
            // Cancel the grab if its action failed to create a node.
            if (nodes.size() <= panel.GrabAfter) {
                panel.Drag = NodeDrag::None;
                return;
            }
            panel.DragNode = plan.ActiveNode;
            panel.MoveStart = r.get<const NodeLayout>(nodes[panel.DragNode].Entity).Position;
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
            if (hit.Type == HitType::Socket || hit.Type == HitType::Link) {
                panel.Drag = NodeDrag::Wire;
                panel.DragNode = hit.Node;
                panel.DragSocket = hit.Socket;
            } else if (hit.Type == HitType::Node || hit.Type == HitType::Parameters) {
                // Shift-click on the selected active node removes it from the selection.
                // Any other click activates and displays the clicked node, and a click without Shift on an unselected node selects it alone.
                const bool is_selected = r.all_of<NodeSelected>(nodes[hit.Node].Entity), active = hit.Node == plan.ActiveNode;
                if (io.KeyShift && is_selected && active) {
                    auto remaining = selected;
                    std::erase(remaining, hit.Node);
                    select(remaining, NoNode);
                } else {
                    // One frame carries one emitted action, so the click's selection carries its display.
                    const bool display = hit.Node != plan.DisplayNode;
                    if (!io.KeyShift && !is_selected) {
                        select({hit.Node}, hit.Node, display);
                    } else if (!active) {
                        auto extended = selected;
                        if (!is_selected) extended.insert(std::ranges::lower_bound(extended, hit.Node), hit.Node);
                        select(extended, hit.Node, display);
                    } else if (display) {
                        action::Emit(action::nodes::SetDisplay{mesh_entity, nodes[hit.Node].Entity});
                    }
                    panel.Drag = NodeDrag::Move;
                    panel.DragNode = hit.Node;
                    panel.Moved = false;
                }
            } else {
                panel.Drag = NodeDrag::Box;
            }
        } else if (IsMouseClicked(ImGuiMouseButton_Right)) {
            open_add(false);
        }
    }
    // Route editing shortcuts to the hovered node canvas.
    if (panel.Drag == NodeDrag::None && panel.Hovered) {
        constexpr auto Key{ImGuiInputFlags_RouteGlobal};
        if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, Key)) {
            open_add(false);
        } else if (Shortcut(ImGuiKey_X, Key) || Shortcut(ImGuiKey_Delete, Key) || Shortcut(ImGuiKey_Backspace, Key)) {
            if (!selected.empty()) action::Emit(action::nodes::DeleteNodes{mesh_entity, EntitiesOf(plan, selected)});
        } else if (const bool replace = Shortcut(ImGuiMod_Shift | ImGuiKey_J, Key); replace || Shortcut(ImGuiKey_J, Key)) {
            panel.PendingLinks = MakeLinks(r, mesh_entity, plan, selected, replace);
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, Key)) {
            if (!selected.empty()) {
                action::Emit(action::nodes::DuplicateNodes{mesh_entity, EntitiesOf(plan, selected)});
                Grab(panel, uint32_t(nodes.size()));
            }
        } else if (Shortcut(ImGuiKey_G, Key)) {
            if (plan.ActiveNode != NoNode && r.all_of<NodeSelected>(nodes[plan.ActiveNode].Entity)) Grab(panel, uint32_t(nodes.size()) - 1u);
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
            if (first && panel.DragNode == plan.ActiveNode) {
                panel.Moved = true;
                panel.MoveStart = r.get<const NodeLayout>(nodes[panel.DragNode].Entity).Position;
            }
            if (!panel.Moved) panel.Drag = NodeDrag::None;
            else if (first || mouse_moved) follow(first);
        }
    }
    if (panel.Drag == NodeDrag::Wire && !IsMouseDown(ImGuiMouseButton_Left)) {
        if (hit.Type == HitType::Socket) DropOnNode(panel, plan, mesh_entity, hit.Node, {hit.Socket, uint16_t(hit.Socket + 1u)});
        else if (hit.Type == HitType::Parameters) DropOnNode(panel, plan, mesh_entity, hit.Node, ParameterRange(nodes[hit.Node].Kind));
        else if (hit.Type == HitType::Node) DropOnNode(panel, plan, mesh_entity, hit.Node, {0u, SocketCount[nodes[hit.Node].Kind]});
        else if (KindSockets(nodes[panel.DragNode].Kind)[panel.DragSocket].Direction == SocketDirection::In && IsLinked(plan, panel.DragNode, panel.DragSocket)) action::Emit(action::nodes::Unlink{mesh_entity, nodes[panel.DragNode].Entity, panel.DragSocket});
        else if (panel.Hovered) open_add(true); // Open Add on an empty-canvas wire drop.
        panel.Drag = NodeDrag::None;
    }
    if (panel.Drag == NodeDrag::Box && !IsMouseDown(ImGuiMouseButton_Left)) {
        const auto press = Im(panel.Press);
        if (ImLengthSqr(mouse - press) >= io.MouseDragThreshold * io.MouseDragThreshold) {
            // Select nodes intersecting the box, extending selection with Shift.
            const ImRect box{ImMin(press, mouse), ImMax(press, mouse)};
            std::vector<uint32_t> banded;
            for (uint32_t n = 0u; n < nodes.size(); ++n) {
                if (box.Overlaps(RectOf(graph, graph.Boxes[n])) || (io.KeyShift && std::ranges::binary_search(selected, n))) banded.push_back(n);
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
// Search titles case-insensitively, placing prefix matches first.
std::vector<uint16_t> SearchKinds(std::string_view text) {
    std::vector<uint16_t> kinds;
    for (uint16_t kind = 0u; kind < KindCount; ++kind) {
        if (Matches(KindTitles[kind], text)) kinds.push_back(kind);
    }
    std::ranges::stable_partition(kinds, [&](uint16_t kind) { return std::ranges::starts_with(std::string_view{KindTitles[kind]}, text, {}, Lower, Lower); });
    return kinds;
}

int MoveHighlight(ImGuiInputTextCallbackData *data) {
    *static_cast<int *>(data->UserData) += data->EventKey == ImGuiKey_DownArrow ? 1 : -1;
    return 0;
}

struct LinkingEntry {
    uint16_t Kind, Socket;
    const char *SocketName;
    std::string Label; // The searched text
};
std::vector<LinkingEntry> LinkingEntries(uint16_t held_kind, uint16_t held_socket) {
    std::vector<LinkingEntry> entries;
    for (uint16_t kind = 0u; kind < KindCount; ++kind) {
        const auto sockets = KindSockets(kind);
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const auto *name = LabelOf(sockets[s]);
            if (Joins(held_kind, held_socket, kind, s)) entries.push_back({kind, s, name, std::string{KindTitles[kind]} + " " + name});
        }
    }
    return entries;
}
constexpr int ResultRows{10};
// Display at most ResultRows rows and scroll to the highlight when it changes.
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

// Add the selected node at the menu position and begin dragging it.
// After dropping a wire, list compatible sockets and connect the new node once it exists.
void DrawAddMenu(NodesPanel &panel, const NodeTheme &theme, float cell, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    if (!BeginPopup("##AddNode")) return;
    const auto &nodes = plan.Nodes;
    const bool linking = panel.Linking && panel.DragNode < nodes.size();
    const auto add = [&](uint16_t kind) {
        if (!interactive) return;
        action::Emit(action::nodes::AddNode{mesh_entity, kind, panel.AddAt});
        Grab(panel, uint32_t(nodes.size()));
    };
    if (IsWindowAppearing()) {
        panel.Search = {};
        panel.SearchHighlight = 0;
        SetKeyboardFocusHere();
    }
    // Use Up and Down for result selection instead of text-history navigation.
    const auto highlight = panel.SearchHighlight;
    SetNextItemWidth(GetFontSize() * 14.f);
    const bool enter = InputTextWithHint("##Search", "Search", panel.Search.data(), panel.Search.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory, MoveHighlight, &panel.SearchHighlight);
    if (IsItemEdited()) panel.SearchHighlight = 0;
    const std::string_view text{panel.Search.data()};
    if (linking) {
        auto entries = LinkingEntries(nodes[panel.DragNode].Kind, panel.DragSocket);
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
            // Place the linking socket at the drop point.
            const auto &entry = entries[*picked];
            const auto socket = SocketCell(entry.Kind, entry.Socket, BoxOf(theme, cell, entry.Kind, false).W, false);
            const uvec2 at{std::max(float(panel.AddAt.x) - socket.x, 0.f), std::max(float(panel.AddAt.y) - socket.y, 0.f)};
            action::Emit(action::nodes::AddNode{mesh_entity, entry.Kind, at}, action::Phase::Stage);
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
    } else {
        const auto found = SearchKinds(text);
        const auto picked = DrawResults(int(found.size()), GetFontSize() * 18.f, panel.SearchHighlight, highlight, enter, [&](int i, bool highlighted) {
            const bool clicked = Selectable(KindTitles[found[i]], highlighted);
            SameLine();
            TextDisabled("%s", GroupNames[uint8_t(KindGroups[found[i]])]);
            return clicked;
        });
        if (picked) add(found[*picked]);
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void DrawPortPicker(NodesPanel &panel, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    if (panel.PickNode == NoNode) return;
    if (!BeginPopup("##PickPort")) {
        panel.PickNode = NoNode;
        return;
    }
    const auto &nodes = plan.Nodes;
    const auto node = panel.PickNode;
    if (node < nodes.size() && panel.DragNode < nodes.size()) {
        const auto sockets = KindSockets(nodes[node].Kind);
        for (const auto s : LinkingSockets(plan, panel.DragNode, panel.DragSocket, node, panel.PickRange)) {
            if (MenuItem(LabelOf(sockets[s])) && interactive) action::Emit(LinkOf(plan, mesh_entity, panel.DragNode, panel.DragSocket, node, s));
        }
    } else {
        CloseCurrentPopup();
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void RecordHover(const state::Registry &r, const NodesPanel &panel, const CookPlan &plan, const Hit &hit) {
    if (hit.Type == HitType::None) return;
    const auto [entity, kind] = plan.Nodes[hit.Node];
    if (hit.Type == HitType::Node) {
        const auto *error = r.try_get<const NodeError>(entity);
        ui::Hovered = {.Label = KindTitles[kind], .Path = GroupNames[uint8_t(KindGroups[kind])], .Type = KindTitles[kind], .Help = KindHelp[kind], .Status = error ? FaultTexts[uint8_t(error->Fault)] : nullptr};
    } else if (hit.Type == HitType::Parameters) {
        ui::Hovered = {.Label = "Parameters", .Path = KindTitles[kind], .Help = "Hover to show the node's parameter sockets. Drop a wire here to link it to a compatible parameter."};
    } else if (hit.Type == HitType::Socket) {
        const auto socket = KindSockets(kind)[hit.Socket];
        const bool held = panel.DragNode == hit.Node && panel.DragSocket == hit.Socket;
        const bool dimmed = panel.Drag == NodeDrag::Wire && !held && !Accepts(plan, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket);
        const bool linked = socket.Direction == SocketDirection::In && IsLinked(plan, hit.Node, hit.Socket);
        ui::Hovered = {
            .Label = LabelOf(socket),
            .Path = KindTitles[kind],
            .Type = SocketTypeText(socket),
            .Help = SocketHelps[SocketFirst[kind] + hit.Socket],
            .Status = dimmed ? "Cannot link with the held wire" : linked ? "Linked" :
                                                                           nullptr,
        };
    } else {
        const auto link = plan.Inputs[plan.InputFirst[hit.Node] + hit.Socket];
        ui::Hovered = {
            .Label = "Link",
            .Path = KindTitles[kind],
            .Type = SocketTypeText(KindSockets(plan.Nodes[link.FromNode].Kind)[link.FromSocket]),
            .Help = "Drag the link's input end. Drop it on an output to relink it, or on empty canvas to unlink it.",
        };
    }
}

void DrawCanvas(const state::Registry &r, FrameState &frame, NodesPanel &panel, const NodeTheme &theme, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    const auto &io = GetIO();
    const auto &nodes = plan.Nodes;
    if ((panel.Drag == NodeDrag::Move || panel.Drag == NodeDrag::Wire || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode)) && panel.DragNode >= nodes.size()) panel.Drag = NodeDrag::None;
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
            // Pan with trackpad scrolling; zoom with Ctrl/Cmd scrolling, the mouse wheel, or a pinch.
            const auto wheel = std::exchange(frame.PreciseWheelDelta, vec2{0});
            const auto magnification = std::exchange(frame.Magnification, 0.f);
            const bool pan = frame.PreciseScrolling && !io.KeyCtrl && !io.KeySuper;
            if (pan) canvas.Offset = vec2{Im(canvas.Offset) + Im(wheel) * TrackpadPoints};
            const float steps = pan ? 0.f : wheel.y * (frame.PreciseScrolling ? TrackpadPoints / TrackpadStepPoints : 1.f);
            // Preserve the canvas point under the cursor while zooming.
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
    // Keep cell zero at or beyond the canvas corner, so no negative cell shows.
    canvas.Offset = {std::min(canvas.Offset.x, 0.f), std::min(canvas.Offset.y, 0.f)};
    const float cell = CellOf(theme);
    const auto graph = GraphOf(r, plan, theme, origin + Im(canvas.Offset), cell, canvas.Zoom, panel.Expanded);
    const auto hit = panel.Hovered ? HitTest(graph, mouse) : Hit{};
    if (panel.Drag == NodeDrag::None || panel.Drag == NodeDrag::Wire) RecordHover(r, panel, plan, hit);
    const auto selected = SelectedNodes(r, plan);
    // A held node inserts into a link while it is the sole selected node.
    const bool holding = (panel.Drag == NodeDrag::Move && panel.Moved) || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode);
    const auto insert = holding && selected == std::vector{panel.DragNode} ? InsertTarget(graph, panel.DragNode, mouse) : std::nullopt;
    if (interactive) Interact(r, panel, mesh_entity, graph, selected, hit, insert);
    // Keep parameters expanded while hovered, wiring, or selecting a port.
    const bool wiring = panel.Drag == NodeDrag::Wire;
    const bool stays_open = graph.Expanded != NoNode && (TrayRect(graph, graph.Expanded).Contains(mouse) || (wiring && panel.DragNode == graph.Expanded) || panel.PickNode == graph.Expanded);
    if (hit.Type == HitType::Parameters) panel.Expanded = hit.Node;
    else if (!stays_open) panel.Expanded = NoNode;
    const auto target = panel.Drag == NodeDrag::None ? std::nullopt : insert;

    auto &draw = *GetWindowDrawList();
    const auto zoom = canvas.Zoom;
    const auto &colors = theme.Colors;
    draw.PushClipRect(origin, end, true);
    draw.AddRectFilled(origin, end, U32(colors.Background));
    // A line at every cell, coarse at every CoarseGridCells cells from cell zero.
    const auto grid = U32(colors.Grid), coarse_grid = U32(colors.CoarseGrid);
    const auto coarse = int32_t(std::max<uint8_t>(theme.Sizes.CoarseGridCells, 1u));
    for (auto i = int32_t(std::floor(-canvas.Offset.x / graph.Step)); graph.Origin.x + float(i) * graph.Step < end.x; ++i) {
        const float x = graph.Origin.x + float(i) * graph.Step;
        draw.AddLine({x, origin.y}, {x, end.y}, i % coarse == 0 ? coarse_grid : grid);
    }
    for (auto i = int32_t(std::floor(-canvas.Offset.y / graph.Step)); graph.Origin.y + float(i) * graph.Step < end.y; ++i) {
        const float y = graph.Origin.y + float(i) * graph.Step;
        draw.AddLine({origin.x, y}, {end.x, y}, i % coarse == 0 ? coarse_grid : grid);
    }
    const float thickness = std::max(theme.Sizes.WireWidth * zoom, 1.f);
    ForEachWire(graph, [&](const Wire &wire, PlanLink link, uint32_t node, uint16_t input) {
        const bool lit = (panel.Drag == NodeDrag::None && hit.Type == HitType::Link && hit.Node == node && hit.Socket == input) || (target && target->ToNode == node && target->ToSocket == input);
        const auto color = lit ? colors.WireHighlight : SocketColor(colors, KindSockets(nodes[link.FromNode].Kind)[link.FromSocket]);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(color), thickness);
    });
    // Draw node text at the zoomed font size.
    PushFont(nullptr, GetStyle().FontSizeBase * zoom);
    const auto font_size = GetFontSize();
    const float inset = (theme.Sizes.SocketRadius + theme.Sizes.LabelSpacing) * zoom, spacing = theme.Sizes.LabelSpacing * zoom, label_reach = float(theme.Sizes.MaxLabelCells) * graph.Step;
    // Draw the label from the box's left, or against its right when `right` fits, with an ellipsis past the box.
    const auto draw_label = [&](ImVec2 min, ImVec2 max, const char *text, vec4 color, bool right) {
        if (const float width = CalcTextSize(text).x; right && width <= max.x - min.x) min.x = max.x - width;
        PushStyleColor(ImGuiCol_Text, ImVec4{color.x, color.y, color.z, color.w});
        RenderTextEllipsis(&draw, min, max, max.x, text, nullptr, nullptr);
        PopStyleColor();
    };
    const auto draw_node = [&](uint32_t n) {
        const auto [entity, kind] = nodes[n];
        const auto [min, max] = RectOf(graph, graph.Boxes[n]);
        const bool expanded = n == graph.Expanded;
        const float rounding = theme.Sizes.Rounding * zoom, pad = theme.Sizes.TitlePadding * zoom;
        draw.AddRectFilled(min, max, U32(expanded ? vec4{colors.Node.x, colors.Node.y, colors.Node.z, 1.f} : colors.Node), rounding);
        draw.AddRectFilled(min, {max.x, min.y + graph.Step}, U32(n == plan.DisplayNode ? colors.DisplayedTitle : colors.*GroupColors[uint8_t(KindGroups[kind])]), rounding, ImDrawFlags_RoundTop);
        draw_label({min.x + pad, min.y + (graph.Step - font_size) * .5f}, {max.x - pad, min.y + (graph.Step + font_size) * .5f}, KindTitles[kind], colors.Title, false);
        const auto sockets = KindSockets(kind);
        const auto [first, end_parameter] = ParameterRange(kind);
        const bool erring = r.all_of<NodeError>(entity);
        // Prioritize active, selected, then error border colors; retain an inner error line on selected nodes with errors.
        // Center borders on the node edge to align them with socket centers.
        const bool active = n == plan.ActiveNode, is_selected = r.all_of<NodeSelected>(entity);
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
            const bool parameter = s >= first && s < end_parameter;
            if (parameter && !expanded) continue;
            const auto at = SocketAt(graph, n, s);
            const bool held = wiring && panel.DragNode == n && panel.DragSocket == s;
            const bool dimmed = wiring && !held && !Accepts(plan, panel.DragNode, panel.DragSocket, n, s);
            const bool hovered = hit.Type == HitType::Socket && hit.Node == n && hit.Socket == s && !dimmed;
            DrawSocket(draw, theme, at, theme.Sizes.SocketRadius * zoom, sockets[s], dimmed ? colors.Dimmed : SocketColor(colors, sockets[s]), hovered);
            const auto color = dimmed ? colors.DimmedLabel : parameter ? colors.ParameterLabel :
                                                                         colors.Label;
            const float top = at.y - font_size * .5f, bottom = at.y + font_size * .5f;
            if (s < InputCount[kind]) draw_label({at.x + inset + (parameter ? theme.Sizes.ParameterIndent * zoom : 0.f), top}, {std::min(at.x + label_reach, max.x) - spacing, bottom}, LabelOf(sockets[s]), color, false);
            else draw_label({std::max(at.x - label_reach, min.x) + spacing, top}, {at.x - inset, bottom}, LabelOf(sockets[s]), color, true);
        }
        if (first < end_parameter && !expanded) {
            const auto at = SocketAt(graph, n, first);
            const auto linked = LinkedParameters(plan, n);
            const bool dimmed = wiring && LinkingSockets(plan, panel.DragNode, panel.DragSocket, n, {first, end_parameter}).empty();
            const bool hovered = hit.Type == HitType::Parameters && hit.Node == n && !dimmed;
            const auto color = dimmed ? colors.Dimmed : hovered ? colors.Hovered :
                linked.empty()                                  ? colors.Chevron :
                                                                  colors.ChevronLinked;
            const float chevron = theme.Sizes.SocketRadius * zoom;
            const ImVec2 a{at.x - chevron * .7f, at.y - chevron}, b{at.x - chevron * .7f, at.y + chevron}, c{at.x + chevron * .9f, at.y};
            draw.AddTriangleFilled(a, c, b, U32(color));
            draw.AddTriangle(a, c, b, U32(colors.Outline), theme.Sizes.OutlineWidth);
            std::string label;
            for (const auto socket : linked) label += std::string{label.empty() ? "" : ", "} + socket.Name;
            if (label.empty()) label = "Parameters";
            draw_label({at.x + inset, at.y - font_size * .5f}, {max.x - spacing, at.y + font_size * .5f}, label.c_str(), color, false);
        }
    };
    for (uint32_t n = 0u; n < nodes.size(); ++n) {
        if (n != graph.Expanded) draw_node(n);
    }
    if (graph.Expanded != NoNode) draw_node(graph.Expanded);
    PopFont();
    if (target) draw.AddCircleFilled(target->At, theme.Sizes.SocketRadius * zoom, U32(colors.InsertMarker));
    if (wiring) {
        const auto socket = SocketAt(graph, panel.DragNode, panel.DragSocket);
        const auto held = KindSockets(nodes[panel.DragNode].Kind)[panel.DragSocket];
        const auto wire = held.Direction == SocketDirection::Out ? WireOf(socket, mouse, theme.Sizes.WireCurving) : WireOf(mouse, socket, theme.Sizes.WireCurving);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(SocketColor(colors, held)), thickness);
    }
    if (panel.Drag == NodeDrag::Box) {
        const auto press = Im(panel.Press);
        const auto box_min = ImMin(press, mouse), box_max = ImMax(press, mouse);
        draw.AddRectFilled(box_min, box_max, U32(colors.BoxSelect));
        draw.AddRect(box_min, box_max, U32(colors.BoxSelectBorder));
    }
    draw.PopClipRect();
    DrawAddMenu(panel, theme, cell, mesh_entity, plan, interactive);
    DrawPortPicker(panel, mesh_entity, plan, interactive);
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

// Commit attribute-name changes through SetAttributeName when the field deactivates.
// Retain text for the single active ImGui field.
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
concept LeafParameter = std::same_as<F, AttributeName> || std::same_as<F, state::Entity> || std::same_as<F, mat4>;

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
            // A matrix shows its rows for reading.
            Status = nullptr;
            const auto &m = ui::ReadChain<Prefix..., M>(Value);
            for (int row = 0; row < 4; ++row) {
                std::array values{m[0][row], m[1][row], m[2][row], m[3][row]};
                PushID(row);
                InputFloat4(row == 0 ? label : "##row", values.data(), "%.3f", ImGuiInputTextFlags_ReadOnly);
                PopID();
            }
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

// Use inline selection for an unlinked selection input; an absent mask selects everything.
void DrawInlineSelection(const state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, uint32_t node, bool interactive) {
    const auto sockets = KindSockets(plan.Nodes[node].Kind);
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
    else if (counts == std::array<uint32_t, 3>{}) TextUnformatted("No elements");
    else Text("%u vertices, %u edges, %u faces", counts[0], counts[1], counts[2]);
    // While the node's input is displayed, the viewport selection is the node's selection.
    const auto input = plan.Inputs[plan.InputFirst[node]];
    bool in_viewport = input.FromNode != NoNode && input.FromNode == plan.DisplayNode && input.FromSocket == ResultSocket(plan.Nodes[input.FromNode].Kind);
    BeginDisabled(input.FromNode == NoNode);
    if (Checkbox("Select in Viewport", &in_viewport) && interactive) action::Emit(action::nodes::SetDisplay{mesh_entity, in_viewport ? plan.Nodes[input.FromNode].Entity : entity});
    EndDisabled();
    ui::Describe("Select in Viewport", "Display the node's input, so selecting elements in the viewport sets the node's selection");
    SameLine();
    BeginDisabled(every);
    if (Button("All") && interactive) action::Emit(action::nodes::SetInlineSelection{entity, std::make_unique<InlineSelection>(InlineSelection{.Domain = selection->Domain, .Active = selection->Active, .All = true})});
    EndDisabled();
    ui::Describe("All", "Set the node's selection to all elements");
}

void DrawParameters(state::Registry &r, NodesPanel &panel, state::Entity mesh_entity, const CookPlan &plan, bool interactive) {
    if (plan.ActiveNode == NoNode) {
        TextWrapped("Right click or press Shift+A to add a node.");
        return;
    }
    const auto node = plan.ActiveNode;
    const auto [active, kind] = plan.Nodes[node];
    PushID(int(state::Integral(active)));
    SeparatorText(KindTitles[kind]);
    const auto linked = LinkedParameters(plan, node);
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
} // namespace

void DrawNodesPanel(state::Registry &r, NodesPanel &panel, bool interactive) {
    const auto mesh_entity = action::nodes::EditedMesh(r);
    const auto *plan = r.try_get<const CookPlan>(mesh_entity);
    if (interactive && panel.LinkAfter != NoNode) {
        // Commit node creation and linking as one gesture, then begin moving the node.
        if (plan && plan->Nodes.size() > panel.LinkAfter && panel.DragNode < plan->Nodes.size()) {
            action::Emit(LinkOf(*plan, mesh_entity, panel.DragNode, panel.DragSocket, panel.LinkAfter, panel.LinkSocket), action::Phase::Stage);
            Grab(panel, panel.LinkAfter);
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
        if (!plan) {
            panel.Drag = NodeDrag::None;
            TextUnformatted("Select a mesh object to edit its geometry procedure.");
        } else {
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
            BeginChild("Canvas", {0.f, 0.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
            PopStyleVar();
            DrawCanvas(r, r.Resources.get<FrameState>(), panel, r.get<const NodeTheme>(state::Document), mesh_entity, *plan, interactive);
            EndChild();
        }
    }
    End();
}

void DrawParametersPanel(state::Registry &r, NodesPanel &nodes, Panel &panel, bool interactive) {
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        const auto mesh_entity = action::nodes::EditedMesh(r);
        const auto *plan = r.try_get<const CookPlan>(mesh_entity);
        if (!plan) TextUnformatted("Select a mesh object to edit its geometry procedure.");
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
