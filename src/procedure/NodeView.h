#pragma once

#include "procedure/Macros.h"
#include "procedure/Procedure.h"

struct NodesPanel;

namespace procedure {
// A node of a page: the item it stands for, its title, its kind's title, and its kind, which is KindCount on a scene page.
struct EditorNode {
    state::Entity Item;
    std::string Title;
    const char *Type{};
    uint16_t Kind{KindCount};
    bool operator==(const EditorNode &) const = default;
};
// A page of a network as the editor shows it: its nodes, each node's sockets, inputs first, and the authored link in each input slot.
// A geometry page shows the nodes of a geometry procedure's, a definition's or a material graph's network, which Authored holds, and a scene page shows items.
struct NodeView {
    state::Entity Network;
    const state::Registry *Authored{}; // The registry holding a geometry page's nodes, or null for a scene page
    std::vector<EditorNode> Nodes;
    std::vector<uint32_t> ByItem; // Node indices by item, then index
    std::vector<uint32_t> InputFirst, SocketFirst; // Each node's first input slot and first socket, then the totals
    std::vector<Socket> Sockets;
    std::vector<uint16_t> SocketIds; // The socket or port id of each socket of a geometry page
    std::vector<std::string> Labels; // Each socket's label
    std::vector<DataTypes> ShownTypes; // The data types each socket shows, which for an op socket are its resolved type
    std::vector<std::optional<uint32_t>> Declarations; // The MaterialX declaration each op node of a geometry page resolves to
    std::vector<PortBinding> Bindings; // What each socket of a scene page refers to
    std::vector<NodeLink> Wires; // By input slot
    uint32_t DisplayNode{NoNode};
    bool operator==(const NodeView &) const = default;
};
inline const state::Registry &AuthoredOf(const state::Registry &r, const NodeView &view) { return view.Authored ? *view.Authored : r; }
inline std::span<const Socket> NodeSockets(const NodeView &view, uint32_t node) { return std::span{view.Sockets}.subspan(view.SocketFirst[node], view.SocketFirst[node + 1] - view.SocketFirst[node]); }
// The node's number of input slots: one per input socket, then one per extra link of its multi-input socket.
inline uint16_t SlotsOf(const NodeView &view, uint32_t node) { return uint16_t(view.InputFirst[node + 1] - view.InputFirst[node]); }
inline uint16_t InputsOf(const NodeView &view, uint32_t node) {
    const auto kind = view.Nodes[node].Kind;
    return kind < KindCount && MultiInput[kind] != NoSocket ? InputCount[kind] : SlotsOf(view, node);
}
inline uint16_t SlotSocket(const NodeView &view, uint32_t node, uint16_t slot) { return slot < InputsOf(view, node) ? slot : MultiInput[view.Nodes[node].Kind]; }
// The slot's index among the links into its socket.
inline uint16_t SlotIndex(const NodeView &view, uint32_t node, uint16_t slot) { return slot < InputsOf(view, node) ? 0u : uint16_t(slot - InputsOf(view, node) + 1u); }
inline uint16_t SlotOf(const NodeView &view, uint32_t node, uint16_t socket, uint16_t index) { return index == 0u ? socket : uint16_t(InputsOf(view, node) + index - 1u); }
inline uint16_t LinksInto(const NodeView &view, uint32_t node, uint16_t socket) {
    if (view.Wires[view.InputFirst[node] + socket].FromNode == NoNode) return 0u;
    return NodeSockets(view, node)[socket].Multi ? uint16_t(1u + SlotsOf(view, node) - InputsOf(view, node)) : 1u;
}
inline bool InTray(const NodeView &view, uint32_t node, uint16_t socket) { return view.Nodes[node].Kind < KindCount && InTray(view.Nodes[node].Kind, socket); }
inline uint16_t SocketId(const NodeView &view, uint32_t node, uint16_t socket) { return view.SocketIds[view.SocketFirst[node] + socket]; }
inline const PortBinding &BindingOf(const NodeView &view, uint32_t node, uint16_t socket) { return view.Bindings[view.SocketFirst[node] + socket]; }
// The view's node for the item, or none.
// Of the two nodes a definition's page shows for the definition, it is the first.
inline uint32_t NodeOf(const NodeView &view, state::Entity item) {
    const auto at = std::ranges::lower_bound(view.ByItem, item, {}, [&](uint32_t n) { return view.Nodes[n].Item; });
    return at != view.ByItem.end() && view.Nodes[*at].Item == item ? *at : NoNode;
}
// Whether the node is the output boundary of a definition's page, which stands for the definition with the input boundary.
inline bool OutputBoundary(const NodeView &view, uint32_t node) { return view.Authored && view.Nodes[node].Item == view.Network && SlotsOf(view, node) > 0u; }
// The geometry page of one network: a geometry procedure's or a material's network, or a definition's network between its input boundary and its output boundary.
// Both boundaries stand for the definition: the input boundary, when the definition has input ports, comes first with them as outputs, and the output boundary, when it has output ports, comes last with them as inputs.
// The display shows on the node of the page that is or encloses the occurrence `shown`, whose path starts at this network.
// The nodes have no titles, and the sockets no labels or resolved types.
NodeView ProjectNetwork(const state::Registry &, state::Entity network, std::span<const state::Entity> shown);
// Whether the node can be inserted into the link in `to_slot`.
// Its main input and output must be unlinked and compatible with the link's two ends, and the result must have no cycle.
inline bool Inserts(const NodeView &view, uint32_t node, uint32_t to_node, uint16_t to_slot) {
    const auto link = view.Wires[view.InputFirst[to_node] + to_slot];
    const auto kind = view.Nodes[node].Kind;
    const auto input = MainSocket(kind, SocketDirection::In), output = MainSocket(kind, SocketDirection::Out);
    if (link.FromNode == NoNode || link.FromNode == node || to_node == node || !input || !output) return false;
    if (view.Wires[view.InputFirst[node] + *input].FromNode != NoNode || std::ranges::contains(view.Wires, NodeLink{node, *output})) return false;
    if (!Links(NodeSockets(view, node)[*output], NodeSockets(view, to_node)[SlotSocket(view, to_node, to_slot)]) || !Links(NodeSockets(view, link.FromNode)[link.FromSocket], NodeSockets(view, node)[*input])) return false;
    const auto all = [](uint32_t) { return true; };
    return !Upstream(view.InputFirst, std::span{view.Wires}, node, all)[to_node] && !Upstream(view.InputFirst, std::span{view.Wires}, link.FromNode, all)[node];
}
// The subnet nodes of the document's geometry networks that use each definition, leaving out those of hierarchy instances' copies of procedures.
struct UseCounts {
    std::vector<std::pair<state::Entity, uint32_t>> User; // By user definition
    std::array<uint32_t, MacroCount> Builtin{}; // By Macros index
    std::vector<state::Entity> Used; // Sorted user definitions that any subnet node uses, including those of hierarchy instances' copies
};
UseCounts CountDefinitionUses(const state::Registry &);
// The uses of the user definition, or of the built-in definition of the name for a null definition.
uint32_t UseCount(const UseCounts &, state::Entity definition, std::string_view builtin = {});
// Whether any subnet node uses the user definition, which keeps the definition from deletion.
bool DefinitionUsed(const UseCounts &, state::Entity definition);
NodeView SceneNodeDefinitions(const state::Registry &, state::Entity network);
// The kind of item, such as "Mesh" or "Physics material".
const char *ItemLabel(const state::Registry &, state::Entity);
void DrawSceneAddMenu(const state::Registry &, state::Entity network);
void DrawSceneNodeMenu(const state::Registry &, NodesPanel &, state::Entity);
void DrawReferenceMenu(const state::Registry &, NodesPanel &, SceneReference);
bool EnterNode(const state::Registry &, NodesPanel &, state::Entity);
} // namespace procedure
