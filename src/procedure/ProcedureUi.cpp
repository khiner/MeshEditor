#include "procedure/ProcedureUi.h"

#include "Field.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Nodes.h"
#include "action/View.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "state/Scene.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/TransformEdit.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace procedure {
using namespace ImGui;
using numeric::vec2;

namespace {
constexpr float NodeSpacing{60.f}; // A node inserted on a link clears the link's reader by this gap.
constexpr float LinkReach{4.f}; // A link takes the mouse within this many pixels of its wire.
constexpr float WireStraightening{3.5f}; // A link's handles straighten as the link nears horizontal, as in Blender.
constexpr float MinZoom{.25f}, MaxZoom{3.f}, PanelWidth{280.f};
constexpr float TrackpadPoints{10.f}; // The platform reports a trackpad's scroll at a tenth of its points.
constexpr float ZoomStep{1.1f}, TrackpadStepPoints{100.f}; // A zoom step is one mouse wheel notch, or 100 points of trackpad scroll.

enum class KindGroup : uint8_t { SourceOutput,
                                 Operators,
                                 Combine,
                                 Attributes,
                                 Values,
                                 Generators,
                                 Instances };
constexpr const char *GroupNames[]{"Source and Output", "Operators", "Combine", "Attributes", "Values", "Generators", "Instances"};
constexpr vec4 NodeThemeColors::*GroupColors[]{
    &NodeThemeColors::SourceOutput,
    &NodeThemeColors::Operators,
    &NodeThemeColors::Combine,
    &NodeThemeColors::Attributes,
    &NodeThemeColors::Values,
    &NodeThemeColors::Generators,
    &NodeThemeColors::Instances,
};
using AttributeKinds = action::TypeList<Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, SetPosition>;
using InstanceKinds = action::TypeList<MeshToPoints, InstanceOnPoints, RealizeInstances>;
template<typename K> constexpr KindGroup GroupOf() {
    if constexpr (std::same_as<K, Source> || std::same_as<K, Output> || std::same_as<K, ObjectInfo>) return KindGroup::SourceOutput;
    else if constexpr (std::same_as<K, Join> || std::same_as<K, Boolean>) return KindGroup::Combine;
    else if constexpr (Generator<K>) return KindGroup::Generators;
    else if constexpr (action::InTypeList<K, AttributeKinds>) return KindGroup::Attributes;
    else if constexpr (ValueKind<K>) return KindGroup::Values;
    else if constexpr (action::InTypeList<K, InstanceKinds>) return KindGroup::Instances;
    else return KindGroup::Operators;
}
// "RecalculateNormals" titles "Recalculate Normals".
template<typename K> constexpr auto Title{field::detail::Spaced<field::fixed_string<state::LeafName<K>().size() + 1>{state::LeafName<K>()}>()};
constexpr auto KindGroups{[]<typename... Ks>(action::TypeList<Ks...>) { return std::array{GroupOf<Ks>()...}; }(NodeKinds{})};
constexpr auto KindTitles{[]<typename... Ks>(action::TypeList<Ks...>) { return std::array{Title<Ks>.c_str()...}; }(NodeKinds{})};

constexpr const char *SocketLabels[]{"Geometry", "Selection", "Attribute", "Instances", "Value"};
// A socket's label is its name, or its type's.
const char *LabelOf(Socket socket) { return socket.Name ? socket.Name : SocketLabels[uint8_t(socket.Type)]; }
// The socket color of each attribute type, where a two-component vector colors as a mix.
constexpr vec4 NodeThemeColors::*TypeColors[]{
    &NodeThemeColors::FloatSocket,
    &NodeThemeColors::MixedSocket,
    &NodeThemeColors::VectorSocket,
    &NodeThemeColors::RotationSocket,
    &NodeThemeColors::IntegerSocket,
    &NodeThemeColors::BooleanSocket,
};
// Geometry and instances color by socket type, and a value, attribute or selection socket by the type it carries, or as a mix of types.
vec4 SocketColor(const NodeThemeColors &colors, Socket socket) {
    if (socket.Type == SocketType::Geometry) return colors.GeometrySocket;
    if (socket.Type == SocketType::Instances) return colors.InstancesSocket;
    return std::has_single_bit(socket.Types) ? colors.*TypeColors[std::countr_zero(socket.Types)] : colors.MixedSocket;
}

ImVec2 Im(vec2 v) { return {v.x, v.y}; }
ImU32 U32(vec4 c) { return ColorConvertFloat4ToU32({c.x, c.y, c.z, c.w}); }
// A socket carrying values per element draws as a diamond, as Blender draws a field, and every other socket as a circle.
// Every port shape has an outline, in the hovered color while hovered.
void DrawSocket(ImDrawList &draw, const NodeTheme &theme, ImVec2 at, float radius, Socket socket, vec4 color, bool hovered) {
    const auto fill = U32(color), outline = U32(hovered ? theme.Colors.Hovered : theme.Colors.Outline);
    if (socket.Type == SocketType::Attribute || socket.Type == SocketType::Selection) {
        const float r = radius * 1.2f;
        draw.AddQuadFilled({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, fill);
        draw.AddQuad({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, outline, theme.OutlineWidth);
        return;
    }
    draw.AddCircleFilled(at, radius, fill);
    draw.AddCircle(at, radius, outline, 0, theme.OutlineWidth);
}
float LengthSquared(ImVec2 v) { return v.x * v.x + v.y * v.y; }

// Maps canvas units to the screen.
struct CanvasView {
    ImVec2 Origin, Offset;
    float Zoom;
    ImVec2 ToScreen(ImVec2 p) const { return Origin + Offset + p * Zoom; }
    ImVec2 ToCanvas(ImVec2 s) const { return (s - Origin - Offset) / Zoom; }
};

// A rectangle on the screen.
struct Rect {
    ImVec2 Min, Max;
};
// A node shows its fixed sockets in rows, inputs down the left edge and outputs down the right.
// A node with parameters has a parameters tray under its fixed inputs: closed, one row with a chevron where its parameter wires meet, and open, a row per parameter socket in the chevron's place.
// The row an input socket sits on, and the rows the inputs take.
uint32_t InputRow(uint16_t kind, uint16_t socket, bool expanded) { return expanded ? socket : std::min<uint32_t>(socket, ParameterRange(kind).first); }
uint32_t InputRows(uint16_t kind, bool expanded) { return expanded ? InputCount[kind] : std::min<uint32_t>(InputCount[kind], ParameterRange(kind).first + 1u); }
// A socket's offset in canvas units from the top left of its node of the kind and width.
ImVec2 SocketOffset(const NodeTheme &theme, uint16_t kind, uint16_t socket, float width, bool expanded) {
    const bool input = socket < InputCount[kind];
    const auto row = float(input ? InputRow(kind, socket, expanded) : socket - InputCount[kind]);
    return {input ? 0.f : width, theme.TitleHeight + (row + .5f) * theme.RowHeight};
}
// The procedure's nodes placed on the screen for one frame, one of them expanded or none.
struct Graph {
    const GeometryProcedure &Procedure;
    const NodeTheme &Theme;
    CanvasView View;
    uint32_t Expanded;
    std::vector<Rect> Rects;
    bool IsExpanded(uint32_t node) const { return node == Expanded; }
    ImVec2 SocketAt(uint32_t node, uint16_t socket) const {
        const auto &rect = Rects[node];
        return rect.Min + SocketOffset(Theme, Procedure.Nodes[node].Kind, socket, (rect.Max.x - rect.Min.x) / View.Zoom, IsExpanded(node)) * View.Zoom;
    }
    // The tray's chevron on the node's edge, in the row under the fixed inputs.
    ImVec2 ParametersAt(uint32_t node) const {
        const auto &rect = Rects[node];
        return {rect.Min.x, rect.Min.y + (Theme.TitleHeight + (float(ParameterRange(Procedure.Nodes[node].Kind).first) + .5f) * Theme.RowHeight) * View.Zoom};
    }
    // The open tray's rounded rectangle on the node's edge around the parameter ports.
    Rect TrayRect(uint32_t node) const {
        const auto [first, end] = ParameterRange(Procedure.Nodes[node].Kind);
        const float reach = Theme.SocketRadius * 1.2f * View.Zoom + 3.f, half = Theme.RowHeight * .5f * View.Zoom;
        const auto at = ParametersAt(node);
        return {{at.x - reach, at.y - half}, {at.x + reach, at.y - half + float(end - first) * Theme.RowHeight * View.Zoom}};
    }
};
bool Contains(const Rect &rect, ImVec2 p) { return p.x >= rect.Min.x && p.y >= rect.Min.y && p.x < rect.Max.x && p.y < rect.Max.y; }
// Each node's rectangle is its layout position, and its layout size grown to fit its title and socket rows.
Graph GraphOf(const state::Scene &r, const GeometryProcedure &procedure, const NodeTheme &theme, const CanvasView &view, uint32_t expanded) {
    // A node index a deletion left stale opens no tray.
    const bool open = expanded < procedure.Nodes.size() && ParameterRange(procedure.Nodes[expanded].Kind).second > ParameterRange(procedure.Nodes[expanded].Kind).first;
    Graph graph{procedure, theme, view, open ? expanded : NoNode, {}};
    graph.Rects.reserve(procedure.Nodes.size());
    for (uint32_t n = 0u; n < procedure.Nodes.size(); ++n) {
        const auto [entity, kind] = procedure.Nodes[n];
        const auto &layout = r.get<const NodeLayout>(entity);
        const uint32_t inputs = InputRows(kind, graph.IsExpanded(n)), outputs = SocketCount[kind] - InputCount[kind];
        const auto rows = float(std::max({inputs, outputs, 1u}));
        const ImVec2 min = Im(layout.Position), size{std::max(layout.Size.x, theme.NodeWidth), std::max(layout.Size.y, theme.TitleHeight + rows * theme.RowHeight + theme.NodePadding)};
        graph.Rects.push_back({view.ToScreen(min), view.ToScreen(min + size)});
    }
    return graph;
}
// Whether the node's input socket reads a link.
bool IsLinked(const GeometryProcedure &procedure, uint32_t node, uint16_t socket) { return procedure.Inputs[InputFirst(procedure, node) + socket].FromNode != NoNode; }
// The node's linked parameter sockets.
std::vector<uint16_t> LinkedParameters(const GeometryProcedure &procedure, uint32_t node) {
    const auto [first, end] = ParameterRange(procedure.Nodes[node].Kind);
    std::vector<uint16_t> linked;
    for (auto s = first; s < end; ++s) {
        if (IsLinked(procedure, node, s)) linked.push_back(s);
    }
    return linked;
}

// A cubic wire leaving its output to the right and entering its input from the left, with handles reaching `curving` of the horizontal span.
using Wire = std::array<ImVec2, 4>;
Wire WireOf(ImVec2 from, ImVec2 to, float curving) {
    const float dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const float straightening = dx > 0.f ? std::min(1.f, dy / dx * WireStraightening) : 1.f;
    const ImVec2 handle{curving * dx * straightening, 0.f};
    return {from, from + handle, to - handle, to};
}
// Visits each linked input as `f(wire, link, node, input)`.
template<typename F> void ForEachWire(const Graph &graph, F &&f) {
    const auto &procedure = graph.Procedure;
    for (uint32_t n = 0u, slot = 0u; n < procedure.Nodes.size(); ++n) {
        for (uint16_t i = 0u; i < InputCount[procedure.Nodes[n].Kind]; ++i, ++slot) {
            if (const auto link = procedure.Inputs[slot]; link.FromNode != NoNode) f(WireOf(graph.SocketAt(link.FromNode, link.FromSocket), graph.SocketAt(n, i), graph.Theme.WireCurving), link, n, i);
        }
    }
}
ImVec2 PointOn(const Wire &w, float t) {
    const float u = 1.f - t;
    return w[0] * (u * u * u) + w[1] * (3.f * u * u * t) + w[2] * (3.f * u * t * t) + w[3] * (t * t * t);
}
// The point of the wire, sampled as a polyline, nearest `p`.
ImVec2 NearestOnWire(const Wire &w, ImVec2 p) {
    constexpr int Segments{24};
    auto nearest = w[0];
    for (int s = 0; s < Segments; ++s) {
        const auto a = PointOn(w, float(s) / Segments), ab = PointOn(w, float(s + 1) / Segments) - a;
        const float length = LengthSquared(ab);
        const float t = length > 0.f ? std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / length, 0.f, 1.f) : 0.f;
        if (const auto q = a + ab * t; LengthSquared(p - q) < LengthSquared(p - nearest)) nearest = q;
    }
    return nearest;
}

enum class HitType : uint8_t { None,
                               Socket,
                               Parameters, // A node's parameters tray header
                               Node,
                               Link };
// What lies under the cursor, where a link names the input it enters.
struct Hit {
    HitType Type{HitType::None};
    uint32_t Node{NoNode};
    uint16_t Socket{};
};
// The expanded node comes first, then sockets before node bodies, the last drawn node first, and links come last.
Hit HitTest(const Graph &graph, ImVec2 mouse) {
    const auto count = uint32_t(graph.Rects.size());
    const float reach = std::max(graph.Theme.SocketRadius * graph.View.Zoom, 4.f) + 3.f;
    const auto near = [&](ImVec2 at) { return LengthSquared(at - mouse) <= reach * reach; };
    const auto inside = [&](uint32_t n) { return Contains(graph.Rects[n], mouse); };
    const auto sockets = [&](uint32_t n) -> std::optional<Hit> {
        const auto kind = graph.Procedure.Nodes[n].Kind;
        const auto [first, end] = ParameterRange(kind);
        for (uint16_t s = 0u; s < SocketCount[kind]; ++s) {
            if (!graph.IsExpanded(n) && s >= first && s < end) continue;
            if (near(graph.SocketAt(n, s))) return Hit{HitType::Socket, n, s};
        }
        if (first < end && !graph.IsExpanded(n) && near(graph.ParametersAt(n))) return Hit{HitType::Parameters, n, first};
        return {};
    };
    if (graph.Expanded != NoNode) {
        if (const auto hit = sockets(graph.Expanded)) return *hit;
        if (inside(graph.Expanded)) return {HitType::Node, graph.Expanded, 0u};
    }
    for (auto n = count; n-- > 0u;) {
        if (const auto hit = sockets(n)) return *hit;
    }
    for (auto n = count; n-- > 0u;) {
        if (inside(n)) return {HitType::Node, n, 0u};
    }
    Hit hit;
    ForEachWire(graph, [&](const Wire &wire, SocketLink, uint32_t node, uint16_t input) {
        if (hit.Type == HitType::None && LengthSquared(NearestOnWire(wire, mouse) - mouse) <= LinkReach * LinkReach) hit = {HitType::Link, node, input};
    });
    return hit;
}

// Whether a wire held at a socket of the held kind links with a socket of the kind: an output into an input, held at either end.
bool Joins(uint16_t held_kind, uint16_t held_socket, uint16_t kind, uint16_t socket) {
    const auto held = KindSockets(held_kind)[held_socket], other = KindSockets(kind)[socket];
    if (held.Direction == other.Direction) return false;
    return held.Direction == SocketDirection::Out ? Links(held, other, kind) : Links(other, held, held_kind);
}
// Whether a wire held at one socket links with the other, on another node.
bool Accepts(const GeometryProcedure &procedure, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    return node != held_node && Joins(procedure.Nodes[held_node].Kind, held_socket, procedure.Nodes[node].Kind, socket);
}
// The sockets in the range of the node that a wire held at another node's socket links with.
std::vector<uint16_t> LinkingSockets(const GeometryProcedure &procedure, uint32_t held_node, uint16_t held_socket, uint32_t node, std::pair<uint16_t, uint16_t> range) {
    std::vector<uint16_t> sockets;
    for (auto s = range.first; s < range.second; ++s) {
        if (Accepts(procedure, held_node, held_socket, node, s)) sockets.push_back(s);
    }
    return sockets;
}
// The link a wire held at one socket makes with another.
action::nodes::Link LinkOf(const GeometryProcedure &procedure, state::Entity mesh_entity, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    const bool from_held = KindSockets(procedure.Nodes[held_node].Kind)[held_socket].Direction == SocketDirection::Out;
    return from_held ? action::nodes::Link{mesh_entity, held_node, held_socket, node, socket} : action::nodes::Link{mesh_entity, node, socket, held_node, held_socket};
}
// Drops the held wire on the node's sockets in the range: it links with the one socket that takes it, or the port picker offers the sockets that do.
void DropOnNode(NodesWindow &window, const GeometryProcedure &procedure, state::Entity mesh_entity, uint32_t node, std::pair<uint16_t, uint16_t> range) {
    const auto sockets = LinkingSockets(procedure, window.DragNode, window.DragSocket, node, range);
    if (sockets.size() == 1u) action::Emit(LinkOf(procedure, mesh_entity, window.DragNode, window.DragSocket, node, sockets.front()));
    if (sockets.size() > 1u) {
        window.PickNode = node;
        window.PickRange = range;
        OpenPopup("##PickPort");
    }
}

// A grab moves the selected nodes with the mouse from the active node's position until a click confirms, and Escape or a right click cancels.
// It holds the active node once the node count grows past `after`.
void Grab(NodesWindow &window, uint32_t after) {
    window.Drag = NodeDrag::Grab;
    window.DragNode = NoNode;
    window.GrabAfter = after;
    window.Press = vec2{GetIO().MousePos};
    window.Moved = false;
}

// Links the node a linking add appended to the socket its dropped wire held, commits the add and the link as one step, and grabs the node.
void ContinueLink(NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure *procedure) {
    if (procedure && procedure->Nodes.size() > window.LinkAfter) {
        action::Emit(LinkOf(*procedure, mesh_entity, window.DragNode, window.DragSocket, window.LinkAfter, window.LinkSocket), action::Phase::Stage);
        Grab(window, window.LinkAfter);
    }
    action::Commit();
    window.LinkAfter = NoNode;
}

// Stages the next link of a Make Link press, and commits the press with its last link.
void ContinueMakeLink(NodesWindow &window) {
    action::Emit(window.PendingLinks.front(), action::Phase::Stage);
    window.PendingLinks.erase(window.PendingLinks.begin());
    if (window.PendingLinks.empty()) action::Commit();
}

// The procedure's selected nodes, ascending.
std::vector<uint32_t> SelectedNodes(const state::Scene &r, const GeometryProcedure &procedure) {
    std::vector<uint32_t> selected;
    for (uint32_t n = 0u; n < procedure.Nodes.size(); ++n) {
        if (r.all_of<NodeSelected>(procedure.Nodes[n].Entity)) selected.push_back(n);
    }
    return selected;
}

// The links Make Link adds between the nodes ordered left to right by their layout, one for each adjacent pair at most.
// A pair links from its left node into its right, or the other way when the right node has no inputs or the left no outputs.
// It links the first input in socket order that is unlinked, or with `replace` linked elsewhere when none is unlinked, from the first output that links into it.
// A pair whose link would close a cycle links nothing.
std::vector<action::nodes::Link> MakeLinks(const state::Scene &r, state::Entity mesh_entity, GeometryProcedure procedure, std::vector<uint32_t> nodes, bool replace) {
    std::ranges::stable_sort(nodes, {}, [&](uint32_t n) { return r.get<const NodeLayout>(procedure.Nodes[n].Entity).Position.x; });
    const auto input_first = InputFirsts(procedure);
    std::vector<action::nodes::Link> links;
    for (size_t i = 0u; i + 1u < nodes.size(); ++i) {
        auto from = nodes[i], to = nodes[i + 1u];
        if (InputCount[procedure.Nodes[to].Kind] == 0u || SocketCount[procedure.Nodes[from].Kind] == InputCount[procedure.Nodes[from].Kind]) std::swap(from, to);
        if (Upstream(procedure, input_first, from, [](uint32_t) { return true; })[to]) continue;
        const auto from_kind = procedure.Nodes[from].Kind, to_kind = procedure.Nodes[to].Kind;
        const auto first_link = [&](bool linked) -> std::optional<action::nodes::Link> {
            for (uint16_t input = 0u; input < InputCount[to_kind]; ++input) {
                const auto current = procedure.Inputs[input_first[to] + input];
                if ((current.FromNode != NoNode) != linked) continue;
                for (uint16_t output = InputCount[from_kind]; output < SocketCount[from_kind]; ++output) {
                    if (Links(KindSockets(from_kind)[output], KindSockets(to_kind)[input], to_kind) && (current.FromNode != from || current.FromSocket != output)) return action::nodes::Link{mesh_entity, from, output, to, input};
                }
            }
            return {};
        };
        auto link = first_link(false);
        if (!link && replace) link = first_link(true);
        if (!link) continue;
        procedure.Inputs[input_first[to] + link->ToSocket] = {from, link->FromSocket};
        links.push_back(*link);
    }
    return links;
}

// A link a held node inserts on: the input it enters, its wire's point nearest the mouse, and how far its reader moves right to clear the node.
struct Insert {
    uint32_t ToNode;
    uint16_t ToSocket;
    ImVec2 At;
    float Shift;
};
// The link with its wire nearest the mouse within reach that the node inserts on, while the node is the one selected node.
std::optional<Insert> InsertTarget(const state::Scene &r, const Graph &graph, uint32_t node, ImVec2 mouse) {
    const auto &procedure = graph.Procedure;
    if (SelectedNodes(r, procedure) != std::vector{node}) return {};
    const auto input_first = InputFirsts(procedure);
    std::optional<Insert> insert;
    float nearest = LinkReach * LinkReach;
    ForEachWire(graph, [&](const Wire &wire, SocketLink, uint32_t to_node, uint16_t input) {
        const auto at = NearestOnWire(wire, mouse);
        if (const float distance = LengthSquared(at - mouse); distance <= nearest && Inserts(procedure, input_first, node, to_node, input)) {
            nearest = distance;
            const float gap = (graph.Rects[to_node].Min.x - graph.Rects[node].Max.x) / graph.View.Zoom;
            insert = Insert{to_node, input, at, std::max(NodeSpacing - gap, 0.f)};
        }
    });
    return insert;
}

// Emits the actions the canvas's presses, drags and releases make.
// A move or grab releasing its node over the insert target inserts the node there.
void Interact(const state::Scene &r, NodesWindow &window, state::Entity mesh_entity, const Graph &graph, const Hit &hit, const std::optional<Insert> &insert) {
    const auto &io = GetIO();
    const auto &procedure = graph.Procedure;
    const auto &nodes = procedure.Nodes;
    const auto mouse = io.MousePos;
    const bool mouse_moved = io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f;
    const auto select = [&](std::vector<uint32_t> selected, uint32_t active) { action::Emit(action::nodes::SelectNodes{mesh_entity, std::move(selected), active}); };
    // The selected nodes follow the mouse from the held node's start.
    const auto follow = [&] {
        const auto position = Im(window.MoveStart) + (mouse - Im(window.Press)) / graph.View.Zoom;
        action::Emit(action::UpdateOf<&NodeLayout::Position>(action::OnSelectedDelta{}, vec2{position}), action::Phase::Stage);
    };
    // A move or grab ends by committing the steps it staged with its insert, or by cancelling them.
    const auto end_move = [&](bool commit) {
        if (window.Moved && commit) {
            if (insert) action::Emit(action::nodes::InsertNode{mesh_entity, window.DragNode, insert->ToNode, insert->ToSocket, insert->Shift}, action::Phase::Stage);
            action::Commit();
        } else if (window.Moved) {
            action::Cancel();
        }
        window.Drag = NodeDrag::None;
    };
    if (window.Drag == NodeDrag::Grab) {
        if (window.DragNode == NoNode) {
            // The grab's action applied by this frame, or it failed and there is nothing to grab.
            if (nodes.size() <= window.GrabAfter || procedure.ActiveNode == NoNode) {
                window.Drag = NodeDrag::None;
                return;
            }
            window.DragNode = procedure.ActiveNode;
            window.MoveStart = r.get<const NodeLayout>(nodes[window.DragNode].Entity).Position;
        }
        if (IsMouseClicked(ImGuiMouseButton_Left)) end_move(true);
        else if (IsKeyPressed(ImGuiKey_Escape) || IsMouseClicked(ImGuiMouseButton_Right)) end_move(false);
        else if (!window.Moved || mouse_moved) {
            window.Moved = true;
            follow();
        }
        return;
    }
    if (window.Drag == NodeDrag::None && window.Hovered) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            window.Press = vec2{mouse};
            if (hit.Type == HitType::Node && IsMouseDoubleClicked(ImGuiMouseButton_Left) && nodes[hit.Node].Kind == KindOf<Output>) {
                if (hit.Node != procedure.OutputNode) action::Emit(action::nodes::SetOutput{mesh_entity, hit.Node});
            } else if (hit.Type == HitType::Socket || hit.Type == HitType::Link) {
                window.Drag = NodeDrag::Wire;
                window.DragNode = hit.Node;
                window.DragSocket = hit.Socket;
            } else if (hit.Type == HitType::Node || hit.Type == HitType::Parameters) {
                // Shift toggles the active node out of the selection, and otherwise extends the selection to the node and makes it active.
                const bool selected = r.all_of<NodeSelected>(nodes[hit.Node].Entity), active = hit.Node == procedure.ActiveNode;
                if (io.KeyShift && selected && active) {
                    auto remaining = SelectedNodes(r, procedure);
                    std::erase(remaining, hit.Node);
                    select(std::move(remaining), NoNode);
                } else {
                    if (io.KeyShift || selected) {
                        auto extended = SelectedNodes(r, procedure);
                        if (!selected) extended.insert(std::ranges::lower_bound(extended, hit.Node), hit.Node);
                        if (!active) select(std::move(extended), hit.Node);
                    } else {
                        select({hit.Node}, hit.Node);
                    }
                    window.Drag = NodeDrag::Move;
                    window.DragNode = hit.Node;
                    window.Moved = false;
                }
            } else {
                window.Drag = NodeDrag::Box;
            }
        } else if (IsMouseClicked(ImGuiMouseButton_Right)) {
            window.AddAt = vec2{graph.View.ToCanvas(mouse)};
            window.Linking = false;
            OpenPopup("##AddNode");
        }
    }
    // The hovered canvas takes the editing shortcuts, which the viewport yields while it is hovered.
    if (window.Drag == NodeDrag::None && window.Hovered) {
        constexpr auto Key{ImGuiInputFlags_RouteGlobal};
        if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, Key)) {
            window.AddAt = vec2{graph.View.ToCanvas(mouse)};
            window.Linking = false;
            OpenPopup("##AddNode");
        } else if (Shortcut(ImGuiKey_X, Key) || Shortcut(ImGuiKey_Delete, Key) || Shortcut(ImGuiKey_Backspace, Key)) {
            if (auto selected = SelectedNodes(r, procedure); !selected.empty()) action::Emit(action::nodes::DeleteNodes{mesh_entity, std::move(selected)});
        } else if (Shortcut(ImGuiKey_J, Key)) {
            window.PendingLinks = MakeLinks(r, mesh_entity, procedure, SelectedNodes(r, procedure), false);
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_J, Key)) {
            window.PendingLinks = MakeLinks(r, mesh_entity, procedure, SelectedNodes(r, procedure), true);
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, Key)) {
            // The copies grab as soon as they exist.
            if (auto selected = SelectedNodes(r, procedure); !selected.empty()) {
                action::Emit(action::nodes::DuplicateNodes{mesh_entity, std::move(selected), {}});
                Grab(window, uint32_t(nodes.size()));
            }
        } else if (Shortcut(ImGuiKey_G, Key)) {
            if (procedure.ActiveNode != NoNode && r.all_of<NodeSelected>(nodes[procedure.ActiveNode].Entity)) Grab(window, uint32_t(nodes.size()) - 1u);
        } else if (Shortcut(ImGuiKey_A, Key)) {
            std::vector<uint32_t> all(nodes.size());
            std::ranges::iota(all, 0u);
            if (SelectedNodes(r, procedure) != all) select(std::move(all), procedure.ActiveNode);
        } else if (Shortcut(ImGuiMod_Alt | ImGuiKey_A, Key)) {
            if (!SelectedNodes(r, procedure).empty() || procedure.ActiveNode != NoNode) select({}, NoNode);
        }
    }
    if (window.Drag == NodeDrag::Move) {
        if (!IsMouseDown(ImGuiMouseButton_Left)) end_move(true);
        else if (window.Moved && IsKeyPressed(ImGuiKey_Escape)) end_move(false);
        else if (IsMouseDragging(ImGuiMouseButton_Left)) {
            // The selected nodes follow the held node, which the press made active.
            const bool first = !window.Moved;
            if (first && window.DragNode == procedure.ActiveNode) {
                window.Moved = true;
                window.MoveStart = r.get<const NodeLayout>(nodes[window.DragNode].Entity).Position;
            }
            if (!window.Moved) window.Drag = NodeDrag::None;
            else if (first || mouse_moved) follow();
        }
    }
    if (window.Drag == NodeDrag::Wire && !IsMouseDown(ImGuiMouseButton_Left)) {
        const auto held = KindSockets(nodes[window.DragNode].Kind)[window.DragSocket];
        if (hit.Type == HitType::Socket) DropOnNode(window, procedure, mesh_entity, hit.Node, {hit.Socket, uint16_t(hit.Socket + 1u)});
        else if (hit.Type == HitType::Parameters) DropOnNode(window, procedure, mesh_entity, hit.Node, ParameterRange(nodes[hit.Node].Kind));
        else if (hit.Type == HitType::Node) DropOnNode(window, procedure, mesh_entity, hit.Node, {0u, SocketCount[nodes[hit.Node].Kind]});
        else if (held.Direction == SocketDirection::In && IsLinked(procedure, window.DragNode, window.DragSocket)) action::Emit(action::nodes::Unlink{mesh_entity, window.DragNode, window.DragSocket});
        else if (window.Hovered) {
            // A wire dropped on empty canvas opens the Add menu to link a new node there.
            window.AddAt = vec2{graph.View.ToCanvas(mouse)};
            window.Linking = true;
            OpenPopup("##AddNode");
        }
        window.Drag = NodeDrag::None;
    }
    if (window.Drag == NodeDrag::Box && !IsMouseDown(ImGuiMouseButton_Left)) {
        const auto selected = SelectedNodes(r, procedure);
        const auto press = Im(window.Press);
        if (LengthSquared(mouse - press) >= io.MouseDragThreshold * io.MouseDragThreshold) {
            // A band selects the nodes it touches, extending the selection with Shift, and the active node stays when it remains selected.
            const ImVec2 lo{std::min(press.x, mouse.x), std::min(press.y, mouse.y)}, hi{std::max(press.x, mouse.x), std::max(press.y, mouse.y)};
            std::vector<uint32_t> banded;
            for (uint32_t n = 0u; n < nodes.size(); ++n) {
                const auto &[min, max] = graph.Rects[n];
                if ((min.x < hi.x && min.y < hi.y && max.x > lo.x && max.y > lo.y) || (io.KeyShift && std::ranges::binary_search(selected, n))) banded.push_back(n);
            }
            const auto active = std::ranges::binary_search(banded, procedure.ActiveNode) ? procedure.ActiveNode : NoNode;
            if (banded != selected || active != procedure.ActiveNode) select(std::move(banded), active);
        } else if (!io.KeyShift && (!selected.empty() || procedure.ActiveNode != NoNode)) {
            select({}, NoNode);
        }
        window.Drag = NodeDrag::None;
    }
}

char Lower(char c) { return char(std::tolower(static_cast<unsigned char>(c))); }
// Whether the text occurs in the label, ignoring case, where empty text occurs in every label.
bool Matches(std::string_view label, std::string_view text) { return text.empty() || !std::ranges::search(label, text, {}, Lower, Lower).empty(); }
// The kinds whose title contains the text ignoring case, those whose title starts with it first, each in kind order.
std::vector<uint16_t> SearchKinds(std::string_view text) {
    std::vector<uint16_t> kinds;
    for (uint16_t kind = 0u; kind < KindCount; ++kind) {
        if (Matches(KindTitles[kind], text)) kinds.push_back(kind);
    }
    std::ranges::stable_partition(kinds, [&](uint16_t kind) { return std::ranges::starts_with(std::string_view{KindTitles[kind]}, text, {}, Lower, Lower); });
    return kinds;
}

// Moves the highlight the field's user data holds down or up a row.
int MoveHighlight(ImGuiInputTextCallbackData *data) {
    *static_cast<int *>(data->UserData) += data->EventKey == ImGuiKey_DownArrow ? 1 : -1;
    return 0;
}

// A socket of a kind that a held wire links with, as a linking menu entry named by its kind and socket.
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
// A search result list shows this many rows at most, and scrolls past them.
constexpr int ResultRows{10};
// Lists the results in a region of the width capped at ResultRows rows, with the highlight clamped to them and scrolled into view once it moves from `previous`.
// `row(i, highlighted)` draws result i's row and returns whether a click picked it, and a click or Enter picks a result with `pick(i)` and closes the popup.
void DrawResults(int count, float width, int &highlight, int previous, bool enter, auto &&row, auto &&pick) {
    highlight = std::max(0, std::min(highlight, count - 1));
    const float rows = float(std::min(count, ResultRows));
    BeginChild("##results", {width, rows > 0.f ? rows * GetTextLineHeightWithSpacing() : 1.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    for (int i = 0; i < count; ++i) {
        // A click in the list region closes the popup itself, since the region is not the popup.
        if (row(i, i == highlight)) {
            pick(i);
            CloseCurrentPopup();
        }
        if (i == highlight && highlight != previous) SetScrollHereY();
    }
    EndChild();
    if (enter && count > 0) {
        pick(highlight);
        CloseCurrentPopup();
    }
}

// Adds a node of a picked kind at the menu's canvas position, from its kind groups or, with search text, from the kinds the text finds.
// The added node grabs.
// A linking menu lists every socket that links with the held socket flat, as Blender's link search does, and stages the add for ContinueLink to link that socket and grab the node.
void DrawAddMenu(NodesWindow &window, const NodeTheme &theme, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    if (!BeginPopup("##AddNode")) return;
    const auto &nodes = procedure.Nodes;
    const bool linking = window.Linking && window.DragNode < nodes.size();
    const auto add = [&](uint16_t kind) {
        if (!interactive) return;
        action::Emit(action::nodes::AddNode{mesh_entity, kind, window.AddAt});
        Grab(window, uint32_t(nodes.size()));
    };
    if (IsWindowAppearing()) {
        window.Search = {};
        window.SearchHighlight = 0;
        SetKeyboardFocusHere();
    }
    // Up and Down move the highlight, which the history callback keeps the field from taking as navigation.
    const auto highlight = window.SearchHighlight;
    SetNextItemWidth(GetFontSize() * 14.f);
    const bool enter = InputTextWithHint("##Search", "Search", window.Search.data(), window.Search.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory, MoveHighlight, &window.SearchHighlight);
    if (IsItemEdited()) window.SearchHighlight = 0;
    const std::string_view text{window.Search.data()};
    if (linking) {
        auto entries = LinkingEntries(nodes[window.DragNode].Kind, window.DragSocket);
        std::erase_if(entries, [&](const LinkingEntry &entry) { return !Matches(entry.Label, text); });
        float width = GetFontSize() * 14.f;
        for (const auto &entry : entries) width = std::max(width, CalcTextSize(entry.Label.c_str()).x + GetFontSize() * 2.f);
        // Each row names its kind dimmed before its socket, as Blender's link search does.
        const auto row = [&](int i, bool highlighted) {
            PushID(i);
            const auto at = GetCursorScreenPos();
            const bool clicked = Selectable("##entry", highlighted);
            PopID();
            SetCursorScreenPos(at);
            TextDisabled("%s >", KindTitles[entries[i].Kind]);
            SameLine();
            TextUnformatted(entries[i].SocketName);
            return clicked;
        };
        DrawResults(int(entries.size()), width, window.SearchHighlight, highlight, enter, row, [&](int i) {
            if (!interactive) return;
            // The linking socket lands on the drop point.
            const auto &entry = entries[i];
            action::Emit(action::nodes::AddNode{mesh_entity, entry.Kind, vec2{Im(window.AddAt) - SocketOffset(theme, entry.Kind, entry.Socket, theme.NodeWidth, false)}}, action::Phase::Stage);
            window.LinkAfter = uint32_t(nodes.size());
            window.LinkSocket = entry.Socket;
        });
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
        const auto row = [&](int i, bool highlighted) {
            const bool clicked = Selectable(KindTitles[found[i]], highlighted);
            SameLine();
            TextDisabled("%s", GroupNames[uint8_t(KindGroups[found[i]])]);
            return clicked;
        };
        DrawResults(int(found.size()), GetFontSize() * 18.f, window.SearchHighlight, highlight, enter, row, [&](int i) { add(found[i]); });
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

// Offers the sockets of the picked node that the held wire links with, and links the pick.
void DrawPortPicker(NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    if (window.PickNode == NoNode) return;
    if (!BeginPopup("##PickPort")) {
        window.PickNode = NoNode;
        return;
    }
    const auto &nodes = procedure.Nodes;
    const auto node = window.PickNode;
    if (node < nodes.size() && window.DragNode < nodes.size()) {
        const auto sockets = KindSockets(nodes[node].Kind);
        for (const auto s : LinkingSockets(procedure, window.DragNode, window.DragSocket, node, window.PickRange)) {
            if (MenuItem(LabelOf(sockets[s])) && interactive) action::Emit(LinkOf(procedure, mesh_entity, window.DragNode, window.DragSocket, node, s));
        }
    } else {
        CloseCurrentPopup();
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void DrawCanvas(const state::Scene &r, FrameState &frame, NodesWindow &window, const NodeTheme &theme, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    const auto &io = GetIO();
    const auto &nodes = procedure.Nodes;
    if ((window.Drag == NodeDrag::Move || window.Drag == NodeDrag::Wire || (window.Drag == NodeDrag::Grab && window.DragNode != NoNode)) && window.DragNode >= nodes.size()) window.Drag = NodeDrag::None;
    const auto origin = GetCursorScreenPos();
    const auto size = GetContentRegionAvail();
    InvisibleButton("##canvas", {std::max(size.x, 1.f), std::max(size.y, 1.f)}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    const auto end = GetItemRectMax();
    const auto mouse = io.MousePos;
    window.Hovered = IsItemHovered() || (IsItemActive() && mouse.x >= origin.x && mouse.y >= origin.y && mouse.x < end.x && mouse.y < end.y);
    SetItemKeyOwner(ImGuiKey_MouseWheelY);
    auto &canvas = window.Canvas;
    if (interactive) {
        if (window.Hovered) {
            // A trackpad scroll pans, or zooms with Ctrl or Cmd held, and a mouse wheel and a pinch zoom.
            const auto wheel = std::exchange(frame.PreciseWheelDelta, vec2{0});
            const auto magnification = std::exchange(frame.Magnification, 0.f);
            const bool pan = frame.PreciseScrolling && !io.KeyCtrl && !io.KeySuper;
            if (pan) canvas.Offset = vec2{Im(canvas.Offset) + Im(wheel) * TrackpadPoints};
            const float steps = pan ? 0.f : wheel.y * (frame.PreciseScrolling ? TrackpadPoints / TrackpadStepPoints : 1.f);
            // The canvas point under the cursor stays under it.
            if (const auto zoom = std::clamp(canvas.Zoom * std::pow(ZoomStep, steps) * (1.f + magnification), MinZoom, MaxZoom); zoom != canvas.Zoom) {
                const auto at = (mouse - origin - Im(canvas.Offset)) / canvas.Zoom;
                canvas.Offset = vec2{mouse - origin - at * zoom};
                canvas.Zoom = zoom;
            }
        }
        if (window.Drag == NodeDrag::None && window.Hovered && IsMouseClicked(ImGuiMouseButton_Middle)) window.Drag = NodeDrag::Pan;
        if (window.Drag == NodeDrag::Pan) {
            if (IsMouseDown(ImGuiMouseButton_Middle)) canvas.Offset = vec2{Im(canvas.Offset) + io.MouseDelta};
            else window.Drag = NodeDrag::None;
        }
    }
    const auto graph = GraphOf(r, procedure, theme, {origin, Im(canvas.Offset), canvas.Zoom}, window.Expanded);
    const auto hit = window.Hovered ? HitTest(graph, mouse) : Hit{};
    const bool holding = (window.Drag == NodeDrag::Move && window.Moved) || (window.Drag == NodeDrag::Grab && window.DragNode != NoNode);
    const auto insert = holding ? InsertTarget(r, graph, window.DragNode, mouse) : std::nullopt;
    if (interactive) Interact(r, window, mesh_entity, graph, hit, insert);
    // The mouse over a node's tray chevron opens the tray, which stays open while the mouse is in the tray's rectangle, the wire it holds is held, or the port picker is its.
    const bool wiring = window.Drag == NodeDrag::Wire;
    const bool stays_open = graph.Expanded != NoNode && (Contains(graph.TrayRect(graph.Expanded), mouse) || (wiring && window.DragNode == graph.Expanded) || window.PickNode == graph.Expanded);
    if (hit.Type == HitType::Parameters) window.Expanded = hit.Node;
    else if (!stays_open) window.Expanded = NoNode;
    // The held node's insert target draws lit, with a marker where the node lands on its wire.
    const auto target = window.Drag == NodeDrag::None ? std::nullopt : insert;

    auto &draw = *GetWindowDrawList();
    const auto zoom = canvas.Zoom;
    const auto &colors = theme.Colors;
    draw.PushClipRect(origin, end, true);
    draw.AddRectFilled(origin, end, U32(colors.Background));
    if (const float step = theme.GridSpacing * zoom; step >= 8.f) {
        const auto grid = U32(colors.Grid);
        for (float x = std::fmod(canvas.Offset.x, step); x < end.x - origin.x; x += step) draw.AddLine({origin.x + x, origin.y}, {origin.x + x, end.y}, grid);
        for (float y = std::fmod(canvas.Offset.y, step); y < end.y - origin.y; y += step) draw.AddLine({origin.x, origin.y + y}, {end.x, origin.y + y}, grid);
    }
    const float thickness = std::max(theme.WireWidth * zoom, 1.f);
    ForEachWire(graph, [&](const Wire &wire, SocketLink link, uint32_t node, uint16_t input) {
        const bool lit = (window.Drag == NodeDrag::None && hit.Type == HitType::Link && hit.Node == node && hit.Socket == input) || (target && target->ToNode == node && target->ToSocket == input);
        const auto color = lit ? colors.WireHighlight : SocketColor(colors, KindSockets(nodes[link.FromNode].Kind)[link.FromSocket]);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(color), thickness);
    });
    const auto font_size = GetFontSize() * zoom;
    const float inset = (theme.SocketRadius + theme.LabelSpacing) * zoom;
    const auto draw_node = [&](uint32_t n) {
        const auto [entity, kind] = nodes[n];
        const auto &[min, max] = graph.Rects[n];
        const bool expanded = graph.IsExpanded(n);
        const float rounding = theme.Rounding * zoom, title_bottom = min.y + theme.TitleHeight * zoom;
        const bool result = n == procedure.OutputNode;
        draw.AddRectFilled(min, max, U32(expanded ? vec4{colors.Node.x, colors.Node.y, colors.Node.z, 1.f} : colors.Node), rounding);
        draw.AddRectFilled(min, {max.x, title_bottom}, U32(result ? colors.OutputNode : colors.*GroupColors[uint8_t(KindGroups[kind])]), rounding, ImDrawFlags_RoundCornersTop);
        const ImVec4 clip{min.x, min.y, max.x, max.y};
        draw.AddText(nullptr, font_size, {min.x + theme.TitlePadding * zoom, min.y + (theme.TitleHeight * zoom - font_size) * .5f}, U32(colors.Title), KindTitles[kind], nullptr, 0.f, &clip);
        const auto sockets = KindSockets(kind);
        const auto [first, end_parameter] = ParameterRange(kind);
        const bool erring = r.all_of<NodeError>(entity);
        // The active node borders in the active color and a selected node in the selected color, an erring node in the error color unless selected, and a selected erring node adds an error line inside its border.
        // The border centers on the node's edge, where the sockets sit, and the sockets draw over it.
        // ImGui strokes a rectangle half a unit inside its bounds, so the stroked bounds grow by that half.
        const bool active = n == procedure.ActiveNode, selected = r.all_of<NodeSelected>(entity);
        const auto error = U32(colors.Error);
        const auto border = active ? U32(colors.Active) : selected ? U32(colors.Selected) :
            erring                                                 ? error :
                                                                     U32(colors.Border);
        constexpr ImVec2 StrokeInset{.5f, .5f};
        draw.AddRect(min - StrokeInset, max + StrokeInset, border, rounding, erring || active || selected ? theme.HighlightBorderWidth : theme.BorderWidth);
        if (erring && (active || selected)) draw.AddRect(min + ImVec2{3.f, 3.f}, max - ImVec2{3.f, 3.f}, error, rounding, 1.f);
        if (expanded) {
            const auto tray = graph.TrayRect(n);
            draw.AddRectFilled(tray.Min, tray.Max, U32(colors.Tray), rounding);
            draw.AddRect(tray.Min, tray.Max, U32(colors.Outline), rounding, theme.OutlineWidth);
        }
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const bool parameter = s >= first && s < end_parameter;
            if (parameter && !expanded) continue;
            const auto at = graph.SocketAt(n, s);
            const bool held = wiring && window.DragNode == n && window.DragSocket == s;
            const bool dimmed = wiring && !held && !Accepts(procedure, window.DragNode, window.DragSocket, n, s);
            const bool hovered = hit.Type == HitType::Socket && hit.Node == n && hit.Socket == s && !dimmed;
            DrawSocket(draw, theme, at, theme.SocketRadius * zoom, sockets[s], dimmed ? colors.Dimmed : SocketColor(colors, sockets[s]), hovered);
            const auto *label = LabelOf(sockets[s]);
            // A parameter row indents its label into the tray.
            const float x = s >= InputCount[kind] ? at.x - inset - GetFont()->CalcTextSizeA(font_size, FLT_MAX, 0.f, label).x : at.x + inset + (parameter ? theme.ParameterIndent * zoom : 0.f);
            draw.AddText(nullptr, font_size, {x, at.y - font_size * .5f}, U32(dimmed ? colors.DimmedLabel : parameter ? colors.ParameterLabel :
                                                                                                                        colors.Label),
                         label, nullptr, 0.f, &clip);
        }
        // A closed tray's chevron is named after its linked parameters.
        if (first < end_parameter && !expanded) {
            const auto at = graph.ParametersAt(n);
            const auto linked = LinkedParameters(procedure, n);
            const bool dimmed = wiring && LinkingSockets(procedure, window.DragNode, window.DragSocket, n, {first, end_parameter}).empty();
            const bool hovered = hit.Type == HitType::Parameters && hit.Node == n && !dimmed;
            const auto color = U32(dimmed ? colors.Dimmed : hovered ? colors.Hovered :
                                       linked.empty()               ? colors.Chevron :
                                                                      colors.ChevronLinked);
            const float chevron = theme.SocketRadius * zoom;
            const ImVec2 a{at.x - chevron * .7f, at.y - chevron}, b{at.x - chevron * .7f, at.y + chevron}, c{at.x + chevron * .9f, at.y};
            draw.AddTriangleFilled(a, b, c, color);
            draw.AddTriangle(a, b, c, U32(colors.Outline), theme.OutlineWidth);
            std::string label;
            for (const auto s : linked) label += std::string{label.empty() ? "" : ", "} + sockets[s].Name;
            if (label.empty()) label = "Parameters";
            draw.AddText(nullptr, font_size, {at.x + inset, at.y - font_size * .5f}, color, label.c_str(), nullptr, 0.f, &clip);
        }
    };
    for (uint32_t n = 0u; n < nodes.size(); ++n) {
        if (!graph.IsExpanded(n)) draw_node(n);
    }
    if (graph.Expanded != NoNode) draw_node(graph.Expanded);
    if (target) draw.AddCircleFilled(target->At, theme.SocketRadius * zoom, U32(colors.InsertMarker));
    if (wiring) {
        const auto socket = graph.SocketAt(window.DragNode, window.DragSocket);
        const auto held = KindSockets(nodes[window.DragNode].Kind)[window.DragSocket];
        const auto wire = held.Direction == SocketDirection::Out ? WireOf(socket, mouse, theme.WireCurving) : WireOf(mouse, socket, theme.WireCurving);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], U32(SocketColor(colors, held)), thickness);
    }
    if (window.Drag == NodeDrag::Box) {
        draw.AddRectFilled(Im(window.Press), mouse, U32(colors.BoxSelect));
        draw.AddRect(Im(window.Press), mouse, U32(colors.BoxSelectBorder));
    }
    draw.PopClipRect();
    DrawAddMenu(window, theme, mesh_entity, procedure, interactive);
    DrawPortPicker(window, mesh_entity, procedure, interactive);
}

// Picks a mesh object by name, or none.
bool ObjectChoice(const state::Scene &r, const char *label, state::Entity &value) {
    bool changed = false;
    const auto current = r.valid(value) ? value : state::Null;
    const auto objects = [&] {
        std::vector<state::Entity> choices{state::Null};
        for (const auto [e, instance] : r.view<const Instance>().each()) {
            if (HasMesh(r, instance.Entity)) choices.push_back(e);
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

// Commits an edited attribute name through SetAttributeName when the field deactivates.
// The window keeps the text of the one field ImGui has active.
static_assert(std::same_as<decltype(NodesWindow::NameEdit), AttributeName>);
void NameField(NodesWindow &window, state::Entity node, const char *label, const AttributeName &name, bool interactive) {
    const auto id = GetID(label);
    if (window.NameEditId != id) window.NameEdit = name;
    InputText(label, window.NameEdit.data(), window.NameEdit.size());
    if (IsItemActive()) window.NameEditId = id;
    if (!IsItemDeactivated()) return;
    window.NameEditId = 0;
    if (IsItemDeactivatedAfterEdit() && interactive) action::Emit(action::nodes::SetAttributeName{node, std::string{window.NameEdit.data()}});
}

// The fields of the node's parameters that read a link, by declaration index.
// The sockets of a node's linked parameters, each naming the field it carries.
using LinkedFields = std::vector<Socket>;
bool ReadsLink(const LinkedFields &linked, uint8_t field, uint8_t member = NoField) {
    return std::ranges::any_of(linked, [&](Socket socket) { return socket.Carries(field, member); });
}

// Edits a nested value of a node's parameters through the value type's own editor, each field staging an Update at its offset in the node's component.
// A member reading a link draws disabled.
template<typename K, typename T>
struct NestedEdit : ui::FieldWidgets<NestedEdit<K, T>> {
    state::Scene &R;
    state::Entity Node;
    const T &Value;
    uint16_t Base;
    uint8_t Field;
    const LinkedFields &Linked;

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool delta_capable = false) {
        using F = action::detail::last_field<Ms...>;
        const auto offset = uint16_t(Base + action::detail::FieldOffset<Ms...>());
        const bool reads_link = ReadsLink(Linked, Field, uint8_t(field::IndexOf<action::detail::last_v<Ms...>>));
        BeginDisabled(reads_link);
        const bool changed = ui::detail::RunField(R, Node, ChannelTarget{}, ui::ReadChain<Ms...>(Value), widget, /*selection=*/false, delta_capable && action::DeltaField<F>, [&](action::Target, const F &v) {
            action::Emit(action::Update<F>{Node, state::Key<K>(), offset, v}, action::Phase::Stage);
        });
        EndDisabled();
        if (reads_link) SetItemTooltip("Linked");
        return changed;
    }
};

// Draws each field of the node's parameters: a name field, a value with the field widget, or a nested value through its type's own editor.
// A field reading a link draws disabled.
template<typename K>
void DrawFields(state::Scene &r, NodesWindow &window, state::Entity node, const K &value, const LinkedFields &linked, bool interactive) {
    field::ForEach(value, [&]<size_t I>(const auto &member, std::integral_constant<size_t, I>) {
        using F = std::remove_cvref_t<decltype(member)>;
        static_assert(I < MaxParameterFields);
        const auto offset = uint16_t(reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&value));
        const char *label = field::Label<K, I>.c_str();
        if constexpr (std::same_as<F, AttributeName>) {
            NameField(window, node, label, member, interactive);
        } else if constexpr (std::same_as<F, state::Entity> || ui::DrawableField<F>) {
            // The change stages as an Update on the node, which a release commits.
            const auto widget = [&](F &v) {
                if constexpr (std::same_as<F, state::Entity>) return ObjectChoice(r, label, v);
                else return ui::DrawField(label, v, Spec<K, field::NameString<K, I>>);
            };
            const bool reads_link = ReadsLink(linked, I);
            BeginDisabled(reads_link);
            ui::detail::RunField(r, node, ChannelTarget{}, member, widget, /*selection=*/false, action::DeltaField<F>, [&](action::Target, const F &v) {
                action::Emit(action::Update<F>{node, state::Key<K>(), offset, v}, action::Phase::Stage);
            });
            EndDisabled();
            if (reads_link) SetItemTooltip("Linked");
        } else if constexpr (ui::HasEditor<F>) {
            SeparatorText(label);
            PushID(label);
            NestedEdit<K, F> edit{{}, r, node, member, offset, uint8_t(I), linked};
            ui::DrawEditor(edit, std::type_identity<F>{});
            PopID();
        } else {
            static_assert(false, "DrawFields: this parameter type has no widget");
        }
    });
}

uint32_t SelectedCount(const std::vector<SelectionBlock> &blocks) {
    uint32_t count = 0u;
    for (const auto &block : blocks) {
        for (const auto word : block) count += uint32_t(std::popcount(word));
    }
    return count;
}

// A node's unlinked selection input selects its inline selection: every element while absent, or the elements its masks set.
// A capture in edit mode replaces it with the drawn record's selection, and a clear returns it to every element.
void DrawInlineSelection(const state::Scene &r, state::Entity viewport, state::Entity mesh_entity, const GeometryProcedure &procedure, uint32_t node, bool interactive) {
    const auto sockets = KindSockets(procedure.Nodes[node].Kind);
    const auto socket = std::ranges::find(sockets, SocketType::Selection, &Socket::Type);
    if (socket == sockets.end()) return;
    SeparatorText("Selection");
    if (procedure.Inputs[InputFirst(procedure, node) + uint32_t(socket - sockets.begin())].FromNode != NoNode) {
        TextDisabled("Linked");
        return;
    }
    const auto entity = procedure.Nodes[node].Entity;
    const auto *selection = r.try_get<const InlineSelection>(entity);
    const bool every = !selection || (selection->Vertices.empty() && selection->Edges.empty() && selection->Faces.empty());
    const auto counts = every ? std::array<uint32_t, 3>{} : std::array{SelectedCount(selection->Vertices), SelectedCount(selection->Edges), SelectedCount(selection->Faces)};
    if (every) TextUnformatted("All elements");
    else if (counts == std::array<uint32_t, 3>{}) TextUnformatted("No elements");
    else Text("%u vertices, %u edges, %u faces", counts[0], counts[1], counts[2]);
    const bool edit_mode = r.get<const Interaction>(viewport).Mode == InteractionMode::Edit;
    BeginDisabled(!edit_mode);
    if (Button("Capture") && interactive) action::Emit(action::nodes::CaptureSelection(r, mesh_entity, entity));
    EndDisabled();
    SetItemTooltip(edit_mode ? "Use the elements selected in the viewport" : "Enter edit mode to select elements");
    SameLine();
    BeginDisabled(every);
    if (Button("Clear") && interactive) action::Emit(action::nodes::SetInlineSelection{entity});
    EndDisabled();
    SetItemTooltip("Use all elements");
}

// Edits the active node's parameter component, inline selection and attribute name.
void DrawParameters(state::Scene &r, state::Entity viewport, NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    if (procedure.ActiveNode == NoNode) {
        TextWrapped("Click a node to edit its parameters.");
        TextWrapped("Right click or press Shift+A to add a node.");
        return;
    }
    const auto node = procedure.ActiveNode;
    const auto [active, kind] = procedure.Nodes[node];
    PushID(int(state::Integral(active)));
    SeparatorText(KindTitles[kind]);
    LinkedFields linked;
    for (const auto s : LinkedParameters(procedure, node)) linked.push_back(KindSockets(kind)[s]);
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K>) DrawFields<K>(r, window, active, r.get<const K>(active), linked, interactive);
    });
    DrawInlineSelection(r, viewport, mesh_entity, procedure, node, interactive);
    if (const auto *error = r.try_get<const NodeError>(active)) {
        constexpr const char *Faults[]{"The selection does not fit this operation.", "A parameter is invalid.", "An attribute input is missing or has the wrong type or domain.", "An input mesh is not a closed surface.", "The boolean produced no geometry.", "Geometry cannot be retrieved from the edited object."};
        const auto color = r.get<const NodeTheme>(viewport).Colors.Error;
        PushStyleColor(ImGuiCol_Text, {color.x, color.y, color.z, color.w});
        TextWrapped("%s", Faults[uint8_t(error->Fault)]);
        PopStyleColor();
    }
    PopID();
}
} // namespace

void DrawNodesWindow(state::Scene &r, state::Entity viewport, NodesWindow &window, bool interactive) {
    const auto mesh_entity = action::nodes::EditedMesh(r);
    const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
    if (interactive && window.LinkAfter != NoNode) ContinueLink(window, mesh_entity, procedure);
    else if (interactive && !window.PendingLinks.empty()) ContinueMakeLink(window);
    window.Hovered = false;
    if (!window.Visible) return;
    if (Begin(window.Name, &window.Visible)) {
        if (!procedure) window.Drag = NodeDrag::None;
        if (mesh_entity == state::Null) {
            TextUnformatted("Select a mesh object to edit its geometry procedure.");
        } else if (!procedure) {
            TextUnformatted("The mesh has no geometry procedure.");
            if (Button("Add Procedure") && interactive) action::Emit(action::nodes::AddProcedure{mesh_entity});
        } else {
            const auto avail = GetContentRegionAvail();
            const float panel_width = std::min(PanelWidth, avail.x * .5f);
            PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
            BeginChild("Canvas", {avail.x - panel_width - GetStyle().ItemSpacing.x, 0.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
            PopStyleVar();
            DrawCanvas(r, r.Context.get<FrameState>(), window, r.get<const NodeTheme>(viewport), mesh_entity, *procedure, interactive);
            EndChild();
            SameLine();
            BeginChild("Parameters", {0.f, 0.f}, ImGuiChildFlags_Borders);
            DrawParameters(r, viewport, window, mesh_entity, *procedure, interactive);
            EndChild();
        }
    }
    End();
}
void DrawNodeStyle(state::Scene &r, state::Entity viewport) {
    using C = NodeThemeColors;
    ui::Edit f{r, viewport};
    if (Button("Reset")) action::Emit(action::view::ResetNodeTheme{});
    SeparatorText("Colors");
    auto c = f.Sub<&NodeTheme::Colors>();
    c.Color<&C::Background>();
    c.Color<&C::Grid>();
    c.Color<&C::Node>();
    c.Color<&C::Tray>();
    c.Color<&C::Title>();
    c.Color<&C::Label>();
    c.Color<&C::ParameterLabel>();
    c.Color<&C::Border>();
    c.Color<&C::Selected>();
    c.Color<&C::Active>();
    c.Color<&C::Error>();
    c.Color<&C::Outline>();
    c.Color<&C::Hovered>();
    c.Color<&C::WireHighlight>();
    c.Color<&C::InsertMarker>();
    c.Color<&C::BoxSelect>();
    c.Color<&C::BoxSelectBorder>();
    c.Color<&C::Dimmed>();
    c.Color<&C::DimmedLabel>();
    c.Color<&C::Chevron>();
    c.Color<&C::ChevronLinked>();
    c.Color<&C::SourceOutput>("Source and output title");
    c.Color<&C::Operators>("Operators title");
    c.Color<&C::Combine>("Combine title");
    c.Color<&C::Attributes>("Attributes title");
    c.Color<&C::Values>("Values title");
    c.Color<&C::Generators>("Generators title");
    c.Color<&C::Instances>("Instances title");
    c.Color<&C::OutputNode>("Output node title");
    c.Color<&C::GeometrySocket>();
    c.Color<&C::InstancesSocket>();
    c.Color<&C::FloatSocket>();
    c.Color<&C::IntegerSocket>();
    c.Color<&C::BooleanSocket>();
    c.Color<&C::VectorSocket>();
    c.Color<&C::RotationSocket>();
    c.Color<&C::MixedSocket>();
    SeparatorText("Sizes");
    f.Slider<&NodeTheme::NodeWidth>(nullptr, 80.f, 400.f, "%.0f");
    f.Slider<&NodeTheme::TitleHeight>(nullptr, 12.f, 48.f, "%.0f");
    f.Slider<&NodeTheme::RowHeight>(nullptr, 12.f, 48.f, "%.0f");
    f.Slider<&NodeTheme::NodePadding>(nullptr, 0.f, 24.f, "%.0f");
    f.Slider<&NodeTheme::TitlePadding>(nullptr, 0.f, 24.f, "%.0f");
    f.Slider<&NodeTheme::LabelSpacing>(nullptr, 0.f, 24.f, "%.0f");
    f.Slider<&NodeTheme::ParameterIndent>(nullptr, 0.f, 24.f, "%.0f");
    f.Slider<&NodeTheme::SocketRadius>(nullptr, 2.f, 12.f, "%.1f");
    f.Slider<&NodeTheme::Rounding>(nullptr, 0.f, 12.f, "%.1f");
    f.Slider<&NodeTheme::WireWidth>(nullptr, .5f, 8.f, "%.1f");
    f.Slider<&NodeTheme::WireCurving>(nullptr, 0.f, 1.f, "%.2f");
    f.Slider<&NodeTheme::GridSpacing>(nullptr, 8.f, 128.f, "%.0f");
    f.Slider<&NodeTheme::BorderWidth>(nullptr, .5f, 4.f, "%.1f");
    f.Slider<&NodeTheme::HighlightBorderWidth>(nullptr, .5f, 6.f, "%.1f");
    f.Slider<&NodeTheme::OutlineWidth>(nullptr, 0.f, 4.f, "%.1f");
}
} // namespace procedure
