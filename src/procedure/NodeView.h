#pragma once

#include "procedure/Procedure.h"

struct NodesPanel;

namespace procedure {
// Transient node definitions consumed by the editor. Geometry sockets borrow the projected cook plan.
struct EditorNode {
    NetworkItem Item;
    std::string Title;
    const char *Type{};
    uint16_t Kind{KindCount};
    std::optional<numeric::uvec2> Position;
};
struct NodeView {
    state::Entity Network;
    const CookPlan *Geometry{};
    std::vector<EditorNode> Nodes;
    std::vector<uint32_t> InputFirst, SocketFirst;
    std::vector<NodeLink> Wires;
    uint32_t DisplayNode{NoNode};
    // Reference socket storage belongs to this projection; bindings address canonical document data.
    std::vector<Socket> Sockets;
    std::vector<std::string> Labels;
    std::vector<PortBinding> Bindings;
    std::vector<std::pair<uint32_t, uint32_t>> Relationships;
};
inline auto WiresOf(const NodeView &view) { return std::span{view.Geometry ? view.Geometry->Wires : view.Wires}; }
inline auto InputsFirst(const NodeView &view) { return std::span{view.Geometry ? view.Geometry->InputFirst : view.InputFirst}; }
inline std::span<const Socket> NodeSockets(const NodeView &view, uint32_t node) {
    return view.Geometry ? NodeSockets(*view.Geometry, node) : std::span{view.Sockets}.subspan(view.SocketFirst[node], view.SocketFirst[node + 1] - view.SocketFirst[node]);
}
inline uint16_t InputsOf(const NodeView &view, uint32_t node) {
    const auto first = InputsFirst(view);
    return uint16_t(first[node + 1] - first[node]);
}
inline auto ParameterRange(const NodeView &view, uint32_t node) { return view.Geometry ? ParameterRange(*view.Geometry, node) : std::pair{InputsOf(view, node), InputsOf(view, node)}; }
inline uint16_t SocketId(const NodeView &view, uint32_t node, uint16_t socket) { return SocketId(*view.Geometry, node, socket); }
inline const PortBinding &BindingOf(const NodeView &view, uint32_t node, uint16_t socket) { return view.Bindings[view.SocketFirst[node] + socket]; }
NodeView SceneNodeDefinitions(const state::Registry &, state::Entity network);
void DrawSceneAddMenu(const state::Registry &, state::Entity network);
void DrawSceneNodeMenu(const state::Registry &, NodesPanel &, NetworkItem);
std::string ReferenceDescription(const state::Registry &, const PortBinding &);
void DrawReferenceMenu(const state::Registry &, state::Entity network, SceneReference);
bool EnterNode(const state::Registry &, NodesPanel &, NetworkItem);
} // namespace procedure
