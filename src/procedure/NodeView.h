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
};
inline auto WiresOf(const NodeView &view) { return std::span{view.Geometry ? view.Geometry->Wires : view.Wires}; }
inline auto InputsFirst(const NodeView &view) { return std::span{view.Geometry ? view.Geometry->InputFirst : view.InputFirst}; }
inline std::span<const Socket> NodeSockets(const NodeView &view, uint32_t node) {
    return view.Geometry ? NodeSockets(*view.Geometry, node) : std::span{view.Sockets}.subspan(view.SocketFirst[node], view.SocketFirst[node + 1] - view.SocketFirst[node]);
}
// The node's number of input slots: one per input socket, then one per extra link of its multi-input socket.
inline uint16_t SlotsOf(const NodeView &view, uint32_t node) {
    const auto first = InputsFirst(view);
    return uint16_t(first[node + 1] - first[node]);
}
inline uint16_t InputsOf(const NodeView &view, uint32_t node) {
    return view.Geometry && view.Nodes[node].Kind != KindOf<Subnet> ? InputCount[view.Nodes[node].Kind] : SlotsOf(view, node);
}
inline uint16_t SlotSocket(const NodeView &view, uint32_t node, uint16_t slot) { return slot < InputsOf(view, node) ? slot : MultiInput[view.Nodes[node].Kind]; }
// The slot's index among the links into its socket.
inline uint16_t SlotIndex(const NodeView &view, uint32_t node, uint16_t slot) { return slot < InputsOf(view, node) ? 0u : uint16_t(slot - InputsOf(view, node) + 1u); }
inline uint16_t SlotOf(const NodeView &view, uint32_t node, uint16_t socket, uint16_t index) { return index == 0u ? socket : uint16_t(InputsOf(view, node) + index - 1u); }
inline uint16_t LinksInto(const NodeView &view, uint32_t node, uint16_t socket) {
    if (WiresOf(view)[InputsFirst(view)[node] + socket].FromNode == NoNode) return 0u;
    return NodeSockets(view, node)[socket].Multi ? uint16_t(1u + SlotsOf(view, node) - InputsOf(view, node)) : 1u;
}
inline bool InTray(const NodeView &view, uint32_t node, uint16_t socket) { return view.Geometry && view.Nodes[node].Kind != KindOf<Subnet> && InTray(view.Nodes[node].Kind, socket); }
inline uint16_t SocketId(const NodeView &view, uint32_t node, uint16_t socket) { return SocketId(*view.Geometry, node, socket); }
inline const PortBinding &BindingOf(const NodeView &view, uint32_t node, uint16_t socket) { return view.Bindings[view.SocketFirst[node] + socket]; }
NodeView SceneNodeDefinitions(const state::Registry &, state::Entity network);
// The kind of item, such as "Mesh" or "Physics material".
const char *ItemLabel(const state::Registry &, NetworkItem);
void DrawSceneAddMenu(const state::Registry &, state::Entity network);
void DrawSceneNodeMenu(const state::Registry &, NodesPanel &, NetworkItem);
void DrawReferenceMenu(const state::Registry &, NodesPanel &, SceneReference);
bool EnterNode(const state::Registry &, NodesPanel &, NetworkItem);
} // namespace procedure
