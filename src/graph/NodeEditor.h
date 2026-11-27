#pragma once

#include "Panel.h"

namespace state {
struct Registry;
} // namespace state

namespace graph {
struct SubgraphRef;
state::Entity EditorGraph(const state::Registry &, const NodesPanel &);
// Shows the graph's page, or the page of the subgraph of the last subgraph node of `path`, whose subgraph nodes lead from the graph to it.
void EnterGraph(NodesPanel &, state::Entity graph, state::Entity object = state::Null, std::vector<state::Entity> path = {});
// Turns the editor to the page of the active node when that node changes.
// A geometry page turns to the newly active object's geometry.
void FollowActiveNode(const state::Registry &, NodesPanel &);
void DrawGraphNavigation(const state::Registry &, NodesPanel &);
// The status of a user subgraph's uses, with the effect of its edits when several share it.
std::string UsesText(uint32_t uses);
// Holds the textures of the image Properties draws this frame, or none, and releases the ones it held once the UI frame before this one completes.
void HoldShownImage(const state::Registry &, NodesPanel::ShownImage &, std::shared_ptr<const void> hold);
// Draws the preview of the material, or of the output of the op node at `path` in its graph, filling the panel's width, or else the status.
void DrawMaterialPreview(state::Registry &, NodesPanel &, state::Entity material, std::span<const state::Entity> path);
// The uses of each subgraph, counted again when the graph revision advances.
const UseCounts &UsesOf(const state::Registry &, NodesPanel &);
// Draws the ports of a subgraph's interface, which a built-in subgraph shows read-only.
void DrawSubgraphInterface(state::Registry &, SubgraphRef, bool interactive);
// Lists the entities and fields the animation has channels for, each entity as a button that reveals it.
void DrawAnimationTargets(const state::Registry &, NodesPanel &, uint32_t animation);
void DrawNodesPanel(state::Registry &, NodesPanel &, bool interactive);
// The page the Nodes panel last drew, in cells from the canvas origin: each node's box, each lane of the automatic layout, and each wire with the nodes it links.
struct CanvasGeometry {
    struct Rect {
        numeric::vec2 Min, Max;
    };
    // A path is its start, then the two control points and the end of each cubic segment.
    struct Wire {
        uint32_t From, To;
        std::vector<numeric::vec2> Path;
    };
    std::vector<Rect> Nodes, Lanes;
    std::vector<Wire> Wires;
};
CanvasGeometry DrawnGeometry(const state::Registry &, const NodesPanel &);
// The title the editor draws on a node of a geometry or material graph.
std::string NodeTitle(const state::Registry &, state::Entity node);
// A socket of a graph node, or a port of a subgraph node or a subgraph, with the label the editor draws beside it.
struct SocketTitle {
    uint16_t Id;
    std::string Label;
};
// Empty for an entity that is no graph node, subgraph node or subgraph.
std::vector<SocketTitle> SocketTitles(const state::Registry &, state::Entity node);
// The links Make Links makes between the selected nodes of the page the panel last drew, replacing linked inputs when `replace` holds.
std::vector<action::nodes::Link> MakeLinksOf(const state::Registry &, const NodesPanel &, bool replace);
// Draws the parameters of the active node of the geometry graph the editor shows, and returns that graph's mesh entity, or null for another page.
state::Entity DrawNodeParameters(state::Registry &, NodesPanel &, bool interactive);
// Draws the geometry node the editor shows as active, then the selection's actions and the active node.
// A geometry page shows the active node only when the node draws that page's mesh.
void DrawPropertiesPanel(state::Registry &, NodesPanel &, Panel &, bool interactive);
void DrawNodeStyle(state::Registry &);
} // namespace graph
