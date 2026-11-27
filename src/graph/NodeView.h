#pragma once

#include "graph/BuiltinSubgraphs.h"
#include "graph/Graph.h"

struct NodesPanel;

namespace graph {
// A node of a page: the item it stands for, its title, its kind's title, and its kind, which is KindCount on a scene page.
struct EditorNode {
    state::Entity Item;
    std::string Title;
    const char *Type{};
    uint16_t Kind{KindCount};
    bool operator==(const EditorNode &) const = default;
};
// A page of a graph as the editor shows it: its nodes, each node's sockets, inputs first, and the authored link in each input slot.
// A geometry page shows the nodes of a geometry graph, a subgraph or a material graph, which Authored holds, and a scene page shows items.
struct NodeView {
    state::Entity Graph;
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
// Of the two nodes a subgraph's page shows for the subgraph, it is the first.
inline uint32_t NodeOf(const NodeView &view, state::Entity item) {
    const auto at = std::ranges::lower_bound(view.ByItem, item, {}, [&](uint32_t n) { return view.Nodes[n].Item; });
    return at != view.ByItem.end() && view.Nodes[*at].Item == item ? *at : NoNode;
}
// Whether the node is the output boundary of a subgraph's page, which stands for the subgraph with the input boundary.
inline bool OutputBoundary(const NodeView &view, uint32_t node) { return view.Authored && view.Nodes[node].Item == view.Graph && SlotsOf(view, node) > 0u; }
// The geometry page of one graph: a geometry graph or a material's graph, or a subgraph between its input boundary and its output boundary.
// Both boundaries stand for the subgraph: the input boundary, when the subgraph has input ports, comes first with them as outputs, and the output boundary, when it has output ports, comes last with them as inputs.
// The display shows on the node of the page that is or encloses the occurrence `shown`, whose path starts at this graph.
// The nodes have no titles, and the sockets no labels or resolved types.
NodeView ProjectGraph(const state::Registry &, state::Entity graph, std::span<const state::Entity> shown);
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
// The subgraph nodes that use each subgraph, with the Instances that use each scene subgraph.
struct UseCounts {
    std::vector<std::pair<state::Entity, uint32_t>> User; // By user subgraph
    std::array<uint32_t, BuiltinSubgraphCount> ByBuiltinSubgraph{}; // By BuiltinSubgraphId
    std::vector<state::Entity> Used; // Sorted user subgraphs that any subgraph node or Instance uses
};
UseCounts CountSubgraphUses(const state::Registry &);
// The subgraph nodes that use the subgraph.
uint32_t UseCount(const UseCounts &, SubgraphId);
// Whether any subgraph node uses the user subgraph, which keeps the subgraph from deletion.
bool SubgraphUsed(const UseCounts &, state::Entity subgraph);
NodeView ProjectItemGraph(const state::Registry &, state::Entity graph);
// The kind of item, such as "Mesh" or "Physics material".
const char *ItemLabel(const state::Registry &, state::Entity);
void DrawSceneAddMenu(const state::Registry &, state::Entity graph);
void DrawSceneNodeMenu(const state::Registry &, NodesPanel &, state::Entity);
void DrawReferenceMenu(const state::Registry &, NodesPanel &, SceneReference);
bool EnterNode(const state::Registry &, NodesPanel &, state::Entity);
} // namespace graph
