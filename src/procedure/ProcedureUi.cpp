#include "procedure/ProcedureUi.h"

#include "Field.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Nodes.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "state/Scene.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
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
// Node geometry in canvas units, which the zoom scales to pixels.
constexpr float NodeWidth = 180.f, TitleHeight = 24.f, RowHeight = 22.f, NodePadding = 6.f, SocketRadius = 5.f;
// A node inserted on a link clears the link's reader by this gap.
constexpr float NodeSpacing = 60.f;
// A link takes the mouse within this many pixels of its wire.
constexpr float LinkReach = 4.f;
// Blender's default link curving: handles reach 0.4 of the horizontal span, straightening as the link nears horizontal.
constexpr float WireCurving = .4f, WireStraightening = 3.5f;
constexpr float MinZoom = .25f, MaxZoom = 3.f, GridSpacing = 32.f, PanelWidth = 280.f;
// The platform reports a trackpad's scroll at a tenth of its points.
constexpr float TrackpadPoints = 10.f;
// A zoom step is one mouse wheel notch, or 100 points of trackpad scroll.
constexpr float ZoomStep = 1.1f, TrackpadStepPoints = 100.f;

enum class KindGroup : uint8_t { SourceOutput,
                                 Operators,
                                 Combine,
                                 Attributes,
                                 Generators,
                                 Instances };
constexpr const char *GroupNames[]{"Source and Output", "Operators", "Combine", "Attributes", "Generators", "Instances"};
constexpr ImU32 GroupColors[]{IM_COL32(120, 60, 60, 255), IM_COL32(40, 100, 90, 255), IM_COL32(40, 90, 110, 255), IM_COL32(80, 60, 130, 255), IM_COL32(50, 110, 60, 255), IM_COL32(40, 80, 140, 255)};
using AttributeKinds = action::TypeList<Position, Normal, Index, NamedAttribute, VectorMath, FloatMath, Compare, BooleanMath, StoreNamedAttribute, SetPosition>;
using InstanceKinds = action::TypeList<MeshToPoints, InstanceOnPoints, RealizeInstances>;
template<typename K> constexpr KindGroup GroupOf() {
    if constexpr (std::same_as<K, Source> || std::same_as<K, Output> || std::same_as<K, ObjectInfo>) return KindGroup::SourceOutput;
    else if constexpr (std::same_as<K, Join> || std::same_as<K, Boolean>) return KindGroup::Combine;
    else if constexpr (Generator<K>) return KindGroup::Generators;
    else if constexpr (action::InTypeList<K, AttributeKinds>) return KindGroup::Attributes;
    else if constexpr (action::InTypeList<K, InstanceKinds>) return KindGroup::Instances;
    else return KindGroup::Operators;
}
// "RecalculateNormals" titles "Recalculate Normals".
template<typename K> constexpr auto Title = field::detail::Spaced<field::fixed_string<state::LeafName<K>().size() + 1>{state::LeafName<K>()}>();
constexpr auto KindGroups = []<typename... Ks>(action::TypeList<Ks...>) { return std::array{GroupOf<Ks>()...}; }(NodeKinds{});
constexpr auto KindTitles = []<typename... Ks>(action::TypeList<Ks...>) { return std::array{Title<Ks>.c_str()...}; }(NodeKinds{});

constexpr const char *SocketLabels[]{"Geometry", "Selection", "Attribute", "Instances"};
constexpr ImU32 SocketColors[]{IM_COL32(0, 214, 163, 255), IM_COL32(204, 166, 214, 255), IM_COL32(99, 99, 199, 255), IM_COL32(59, 140, 220, 255)};
ImU32 SocketColor(Socket socket, bool dimmed = false) { return (SocketColors[uint8_t(socket.Type)] & ~IM_COL32_A_MASK) | IM_COL32(0, 0, 0, dimmed ? 60 : 255); }

ImVec2 Im(vec2 v) { return {v.x, v.y}; }
float LengthSquared(ImVec2 v) { return v.x * v.x + v.y * v.y; }

// Maps canvas units to the screen.
struct CanvasView {
    ImVec2 Origin, Offset;
    float Zoom;
    ImVec2 ToScreen(ImVec2 p) const { return Origin + Offset + p * Zoom; }
    ImVec2 ToCanvas(ImVec2 s) const { return (s - Origin - Offset) / Zoom; }
};

// A node's rectangle on the screen.
struct NodeRect {
    ImVec2 Min, Max;
};
// A socket's offset in canvas units from the top left of its node of the kind and width.
// Inputs run down the left edge and outputs down the right, one row each.
ImVec2 SocketOffset(uint16_t kind, uint16_t socket, float width) {
    const bool input = socket < InputCount[kind];
    const auto row = float(input ? socket : socket - InputCount[kind]);
    return {input ? 0.f : width, TitleHeight + (row + .5f) * RowHeight};
}
// The procedure's nodes placed on the screen for one frame.
struct Graph {
    const GeometryProcedure &Procedure;
    CanvasView View;
    std::vector<NodeRect> Rects;
    ImVec2 SocketAt(uint32_t node, uint16_t socket) const {
        const auto &rect = Rects[node];
        return rect.Min + SocketOffset(Procedure.Nodes[node].Kind, socket, (rect.Max.x - rect.Min.x) / View.Zoom) * View.Zoom;
    }
};
// Each node's rectangle is its layout position, and its layout size grown to fit its title and socket rows.
Graph GraphOf(const state::Scene &r, const GeometryProcedure &procedure, const CanvasView &view) {
    Graph graph{procedure, view, {}};
    graph.Rects.reserve(procedure.Nodes.size());
    for (const auto &[entity, kind] : procedure.Nodes) {
        const auto &layout = r.get<const NodeLayout>(entity);
        const uint32_t inputs = InputCount[kind], outputs = SocketCount[kind] - inputs;
        const auto rows = float(std::max({inputs, outputs, 1u}));
        const ImVec2 min = Im(layout.Position), size{std::max(layout.Size.x, NodeWidth), std::max(layout.Size.y, TitleHeight + rows * RowHeight + NodePadding)};
        graph.Rects.push_back({view.ToScreen(min), view.ToScreen(min + size)});
    }
    return graph;
}

// A cubic wire leaving its output to the right and entering its input from the left.
using Wire = std::array<ImVec2, 4>;
Wire WireOf(ImVec2 from, ImVec2 to) {
    const float dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const float straightening = dx > 0.f ? std::min(1.f, dy / dx * WireStraightening) : 1.f;
    const ImVec2 handle{WireCurving * dx * straightening, 0.f};
    return {from, from + handle, to - handle, to};
}
// Visits each linked input as `f(wire, link, node, input)`.
template<typename F> void ForEachWire(const Graph &graph, F &&f) {
    const auto &procedure = graph.Procedure;
    for (uint32_t n = 0u, slot = 0u; n < procedure.Nodes.size(); ++n) {
        for (uint16_t i = 0u; i < InputCount[procedure.Nodes[n].Kind]; ++i, ++slot) {
            if (const auto link = procedure.Inputs[slot]; link.FromNode != NoNode) f(WireOf(graph.SocketAt(link.FromNode, link.FromSocket), graph.SocketAt(n, i)), link, n, i);
        }
    }
}
ImVec2 PointOn(const Wire &w, float t) {
    const float u = 1.f - t;
    return w[0] * (u * u * u) + w[1] * (3.f * u * u * t) + w[2] * (3.f * u * t * t) + w[3] * (t * t * t);
}
// The point of the wire, sampled as a polyline, nearest `p`.
ImVec2 NearestOnWire(const Wire &w, ImVec2 p) {
    constexpr int Segments = 24;
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
                               Node,
                               Link };
// What lies under the cursor, where a link names the input it enters.
struct Hit {
    HitType Type{HitType::None};
    uint32_t Node{NoNode};
    uint16_t Socket{};
};
// Sockets come before node bodies, the last drawn node first, and links come last.
Hit HitTest(const Graph &graph, ImVec2 mouse) {
    const auto count = uint32_t(graph.Rects.size());
    const float reach = std::max(SocketRadius * graph.View.Zoom, 4.f) + 3.f;
    for (auto n = count; n-- > 0u;) {
        for (uint16_t s = 0u; s < SocketCount[graph.Procedure.Nodes[n].Kind]; ++s) {
            if (LengthSquared(graph.SocketAt(n, s) - mouse) <= reach * reach) return {HitType::Socket, n, s};
        }
    }
    for (auto n = count; n-- > 0u;) {
        const auto &[min, max] = graph.Rects[n];
        if (mouse.x >= min.x && mouse.y >= min.y && mouse.x < max.x && mouse.y < max.y) return {HitType::Node, n, 0u};
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
// The first socket of a node of the kind that a wire held at a socket of the held kind links with, or none.
std::optional<uint16_t> LinkingSocket(uint16_t held_kind, uint16_t held_socket, uint16_t kind) {
    for (uint16_t s = 0u; s < SocketCount[kind]; ++s) {
        if (Joins(held_kind, held_socket, kind, s)) return s;
    }
    return {};
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
        using action::nodes::Link;
        const auto node = window.LinkAfter, held_node = window.DragNode;
        const auto held_kind = procedure->Nodes[held_node].Kind, held_socket = window.DragSocket;
        const auto socket = *LinkingSocket(held_kind, held_socket, procedure->Nodes[node].Kind);
        const bool from_held = KindSockets(held_kind)[held_socket].Direction == SocketDirection::Out;
        action::Emit(from_held ? Link{mesh_entity, held_node, held_socket, node, socket} : Link{mesh_entity, node, socket, held_node, held_socket}, action::Phase::Stage);
        Grab(window, node);
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
            } else if (hit.Type == HitType::Node) {
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
        constexpr auto Key = ImGuiInputFlags_RouteGlobal;
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
            std::iota(all.begin(), all.end(), 0u);
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
        if (hit.Type == HitType::Socket) {
            if (Accepts(procedure, window.DragNode, window.DragSocket, hit.Node, hit.Socket)) {
                using action::nodes::Link;
                action::Emit(held.Direction == SocketDirection::Out ? Link{mesh_entity, window.DragNode, window.DragSocket, hit.Node, hit.Socket} : Link{mesh_entity, hit.Node, hit.Socket, window.DragNode, window.DragSocket});
            }
        } else if (hit.Type != HitType::Node && held.Direction == SocketDirection::In && procedure.Inputs[InputFirst(procedure, window.DragNode) + window.DragSocket].FromNode != NoNode) {
            action::Emit(action::nodes::Unlink{mesh_entity, window.DragNode, window.DragSocket});
        } else if (hit.Type != HitType::Node && window.Hovered) {
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

// The kinds whose title contains the text ignoring case, those whose title starts with it first, each in the order given.
std::vector<uint16_t> SearchKinds(std::vector<uint16_t> kinds, std::string_view text) {
    const auto lower = [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); };
    std::erase_if(kinds, [&](uint16_t kind) { return std::ranges::search(std::string_view{KindTitles[kind]}, text, {}, lower, lower).empty(); });
    std::ranges::stable_partition(kinds, [&](uint16_t kind) { return std::ranges::starts_with(std::string_view{KindTitles[kind]}, text, {}, lower, lower); });
    return kinds;
}

// Moves the highlight the field's user data holds down or up a row.
int MoveHighlight(ImGuiInputTextCallbackData *data) {
    *static_cast<int *>(data->UserData) += data->EventKey == ImGuiKey_DownArrow ? 1 : -1;
    return 0;
}

// Adds a node of a picked kind at the menu's canvas position, from its kind groups or, with search text, from the kinds the text finds.
// The added node grabs, and a linking menu offers the kinds that link with the held socket and stages the add for ContinueLink to link and grab it.
void DrawAddMenu(NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    if (!BeginPopup("##AddNode")) return;
    const auto &nodes = procedure.Nodes;
    const bool linking = window.Linking && window.DragNode < nodes.size();
    std::vector<uint16_t> kinds;
    for (uint16_t kind = 0u; kind < KindCount; ++kind) {
        if (!linking || LinkingSocket(nodes[window.DragNode].Kind, window.DragSocket, kind)) kinds.push_back(kind);
    }
    const auto add = [&](uint16_t kind) {
        if (!interactive) return;
        if (!linking) {
            action::Emit(action::nodes::AddNode{mesh_entity, kind, window.AddAt});
            return Grab(window, uint32_t(nodes.size()));
        }
        // The linking socket lands on the drop point.
        const auto socket = *LinkingSocket(nodes[window.DragNode].Kind, window.DragSocket, kind);
        action::Emit(action::nodes::AddNode{mesh_entity, kind, vec2{Im(window.AddAt) - SocketOffset(kind, socket, NodeWidth)}}, action::Phase::Stage);
        window.LinkAfter = uint32_t(nodes.size());
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
    if (const std::string_view text{window.Search.data()}; text.empty()) {
        for (uint8_t group = 0u; group < uint8_t(std::size(GroupNames)); ++group) {
            if (!std::ranges::contains(kinds, KindGroup(group), [](uint16_t kind) { return KindGroups[kind]; }) || !BeginMenu(GroupNames[group])) continue;
            for (const auto kind : kinds) {
                if (KindGroups[kind] == KindGroup(group) && MenuItem(KindTitles[kind])) add(kind);
            }
            EndMenu();
        }
    } else {
        const auto found = SearchKinds(std::move(kinds), text);
        window.SearchHighlight = std::max(0, std::min(window.SearchHighlight, int(found.size()) - 1));
        for (int i = 0; i < int(found.size()); ++i) {
            const auto kind = found[i];
            const bool highlighted = i == window.SearchHighlight;
            if (Selectable(KindTitles[kind], highlighted)) add(kind);
            if (highlighted && window.SearchHighlight != highlight) SetScrollHereY();
            SameLine();
            TextDisabled("%s", GroupNames[uint8_t(KindGroups[kind])]);
        }
        if (enter && !found.empty()) {
            add(found[window.SearchHighlight]);
            CloseCurrentPopup();
        }
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void DrawCanvas(const state::Scene &r, FrameState &frame, NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
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
    const auto graph = GraphOf(r, procedure, {origin, Im(canvas.Offset), canvas.Zoom});
    const auto hit = window.Hovered ? HitTest(graph, mouse) : Hit{};
    const bool holding = (window.Drag == NodeDrag::Move && window.Moved) || (window.Drag == NodeDrag::Grab && window.DragNode != NoNode);
    const auto insert = holding ? InsertTarget(r, graph, window.DragNode, mouse) : std::nullopt;
    if (interactive) Interact(r, window, mesh_entity, graph, hit, insert);
    // The held node's insert target draws lit, with a marker where the node lands on its wire.
    const auto target = window.Drag == NodeDrag::None ? std::nullopt : insert;

    auto &draw = *GetWindowDrawList();
    const auto zoom = canvas.Zoom;
    draw.PushClipRect(origin, end, true);
    draw.AddRectFilled(origin, end, IM_COL32(30, 30, 33, 255));
    if (const float step = GridSpacing * zoom; step >= 8.f) {
        for (float x = std::fmod(canvas.Offset.x, step); x < end.x - origin.x; x += step) draw.AddLine({origin.x + x, origin.y}, {origin.x + x, end.y}, IM_COL32(255, 255, 255, 12));
        for (float y = std::fmod(canvas.Offset.y, step); y < end.y - origin.y; y += step) draw.AddLine({origin.x, origin.y + y}, {end.x, origin.y + y}, IM_COL32(255, 255, 255, 12));
    }
    const float thickness = std::max(2.f * zoom, 1.f);
    ForEachWire(graph, [&](const Wire &wire, SocketLink link, uint32_t node, uint16_t input) {
        const bool lit = (window.Drag == NodeDrag::None && hit.Type == HitType::Link && hit.Node == node && hit.Socket == input) || (target && target->ToNode == node && target->ToSocket == input);
        const auto color = lit ? IM_COL32_WHITE : SocketColor(KindSockets(nodes[link.FromNode].Kind)[link.FromSocket]);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], color, lit ? thickness * 1.5f : thickness);
    });
    const auto font_size = GetFontSize() * zoom;
    const bool wiring = window.Drag == NodeDrag::Wire;
    for (uint32_t n = 0u; n < nodes.size(); ++n) {
        const auto [entity, kind] = nodes[n];
        const auto &[min, max] = graph.Rects[n];
        const float rounding = 4.f * zoom, title_bottom = min.y + TitleHeight * zoom;
        const bool result = n == procedure.OutputNode;
        draw.AddRectFilled(min, max, IM_COL32(48, 48, 52, 240), rounding);
        draw.AddRectFilled(min, {max.x, title_bottom}, result ? IM_COL32(170, 80, 70, 255) : GroupColors[uint8_t(KindGroups[kind])], rounding, ImDrawFlags_RoundCornersTop);
        const ImVec4 clip{min.x, min.y, max.x, max.y};
        draw.AddText(nullptr, font_size, {min.x + 6.f * zoom, min.y + (TitleHeight * zoom - font_size) * .5f}, IM_COL32(235, 235, 235, 255), KindTitles[kind], nullptr, 0.f, &clip);
        const auto sockets = KindSockets(kind);
        const bool erring = r.all_of<NodeError>(entity);
        // The active node borders white and a selected node orange, an erring node red unless selected, and a selected erring node adds a red line inside its border.
        // The sockets draw over the border.
        const bool active = n == procedure.ActiveNode, selected = r.all_of<NodeSelected>(entity);
        constexpr ImU32 Error = IM_COL32(230, 60, 50, 255);
        const auto border = active ? IM_COL32(255, 255, 255, 255) : selected ? IM_COL32(240, 150, 40, 255) :
            erring                                                           ? Error :
                                                                               IM_COL32(15, 15, 15, 255);
        draw.AddRect(min, max, border, rounding, ImDrawFlags_None, erring || active || selected ? 2.f : 1.f);
        if (erring && (active || selected)) draw.AddRect(min + ImVec2{3.f, 3.f}, max - ImVec2{3.f, 3.f}, Error, rounding, ImDrawFlags_None, 1.f);
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const auto at = graph.SocketAt(n, s);
            const bool held = wiring && window.DragNode == n && window.DragSocket == s;
            const bool dimmed = wiring && !held && !Accepts(procedure, window.DragNode, window.DragSocket, n, s);
            draw.AddCircleFilled(at, SocketRadius * zoom, SocketColor(sockets[s], dimmed));
            if (hit.Type == HitType::Socket && hit.Node == n && hit.Socket == s && !dimmed) draw.AddCircle(at, SocketRadius * zoom + 2.f, IM_COL32_WHITE, 0, 1.5f);
            const auto *label = sockets[s].Name ? sockets[s].Name : SocketLabels[uint8_t(sockets[s].Type)];
            const float inset = (SocketRadius + 4.f) * zoom, label_y = at.y - font_size * .5f;
            const float x = s < InputCount[kind] ? at.x + inset : at.x - inset - GetFont()->CalcTextSizeA(font_size, FLT_MAX, 0.f, label).x;
            draw.AddText(nullptr, font_size, {x, label_y}, dimmed ? IM_COL32(200, 200, 200, 60) : IM_COL32(200, 200, 200, 255), label, nullptr, 0.f, &clip);
        }
    }
    if (target) draw.AddCircleFilled(target->At, SocketRadius * zoom, IM_COL32(255, 255, 255, 80));
    if (wiring) {
        const auto socket = graph.SocketAt(window.DragNode, window.DragSocket);
        const auto held = KindSockets(nodes[window.DragNode].Kind)[window.DragSocket];
        const auto wire = held.Direction == SocketDirection::Out ? WireOf(socket, mouse) : WireOf(mouse, socket);
        draw.AddBezierCubic(wire[0], wire[1], wire[2], wire[3], SocketColor(held), thickness);
    }
    if (window.Drag == NodeDrag::Box) {
        draw.AddRectFilled(Im(window.Press), mouse, IM_COL32(100, 140, 220, 40));
        draw.AddRect(Im(window.Press), mouse, IM_COL32(100, 140, 220, 160));
    }
    draw.PopClipRect();
    DrawAddMenu(window, mesh_entity, procedure, interactive);
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

// Draws each field of the node's parameters, a nested struct's fields under its label.
// A field without bounds of its own takes its enclosing field's.
template<typename K, typename T>
void DrawFields(state::Scene &r, NodesWindow &window, state::Entity node, const T &value, uint16_t base, const FieldSpec &enclosing, bool interactive) {
    field::ForEach(value, [&]<size_t I>(const auto &member, std::integral_constant<size_t, I>) {
        using F = std::remove_cvref_t<decltype(member)>;
        const auto offset = uint16_t(base + (reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&value)));
        const char *label = field::Label<T, I>.c_str();
        FieldSpec spec = Spec<T, field::NameString<T, I>>;
        if (!spec.Bounded()) {
            spec.Min = enclosing.Min;
            spec.Max = enclosing.Max;
        }
        if constexpr (std::same_as<F, AttributeName>) {
            NameField(window, node, label, member, interactive);
        } else if constexpr (std::same_as<F, state::Entity> || ui::DrawableField<F>) {
            // The change stages as an Update on the node, which a release commits.
            const auto widget = [&](F &v) {
                if constexpr (std::same_as<F, state::Entity>) return ObjectChoice(r, label, v);
                else return ui::DrawField(label, v, spec);
            };
            ui::detail::RunField(r, node, ChannelTarget{}, member, widget, /*selection=*/false, action::DeltaField<F>, [&](action::Target, const F &v) {
                action::Emit(action::Update<F>{node, state::Key<K>(), offset, v}, action::Phase::Stage);
            });
        } else if constexpr (field::Walkable<F>) {
            SeparatorText(label);
            PushID(label);
            DrawFields<K>(r, window, node, member, offset, spec, interactive);
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
    if (every) TextUnformatted("Every element");
    else if (counts == std::array<uint32_t, 3>{}) TextUnformatted("No elements selected");
    else Text("%u vertices, %u edges, %u faces", counts[0], counts[1], counts[2]);
    const bool edit_mode = r.get<const Interaction>(viewport).Mode == InteractionMode::Edit;
    BeginDisabled(!edit_mode);
    if (Button("Capture") && interactive) action::Emit(action::nodes::CaptureSelection(r, mesh_entity, entity));
    EndDisabled();
    SetItemTooltip(edit_mode ? "Use the elements selected in the viewport" : "Select elements in edit mode");
    SameLine();
    BeginDisabled(every);
    if (Button("Clear") && interactive) action::Emit(action::nodes::SetInlineSelection{entity});
    EndDisabled();
    SetItemTooltip("Use every element");
}

// Edits the active node's parameter component, inline selection and attribute name.
void DrawParameters(state::Scene &r, state::Entity viewport, NodesWindow &window, state::Entity mesh_entity, const GeometryProcedure &procedure, bool interactive) {
    if (procedure.ActiveNode == NoNode) {
        TextWrapped("Click a node to edit its parameters.");
        TextWrapped("Right click or Shift+A adds a node.");
        return;
    }
    const auto node = procedure.ActiveNode;
    const auto [active, kind] = procedure.Nodes[node];
    PushID(int(state::Integral(active)));
    SeparatorText(KindTitles[kind]);
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K>) DrawFields<K>(r, window, active, r.get<const K>(active), 0u, FieldSpec{}, interactive);
    });
    DrawInlineSelection(r, viewport, mesh_entity, procedure, node, interactive);
    if (const auto *error = r.try_get<const NodeError>(active)) {
        constexpr const char *Faults[]{"A geometry input is not connected.", "The selection does not fit this operation.", "A parameter is invalid.", "An attribute input is missing or has the wrong type.", "Both meshes must be closed surfaces.", "The boolean produced no geometry.", "Geometry cannot be retrieved from the edited object."};
        PushStyleColor(ImGuiCol_Text, {.9f, .3f, .25f, 1.f});
        TextWrapped("%s", kind == KindOf<Output> ? "Nothing is connected." : Faults[uint8_t(error->Fault)]);
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
            DrawCanvas(r, r.Context.get<FrameState>(), window, mesh_entity, *procedure, interactive);
            EndChild();
            SameLine();
            BeginChild("Parameters", {0.f, 0.f}, ImGuiChildFlags_Borders);
            DrawParameters(r, viewport, window, mesh_entity, *procedure, interactive);
            EndChild();
        }
    }
    End();
}
} // namespace procedure
