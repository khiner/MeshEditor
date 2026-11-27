#pragma once

#include "graph/GraphSchema.h"
#include "numeric/uvec2.h"
#include "state/DirtySet.h"
#include "state/Entity.h"
#include <algorithm>
#include <compare>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace state {
struct Registry;
}

namespace graph {
// The output socket an input reads, or a null node when unlinked.
struct SocketLink {
    state::Entity FromNode{state::Null};
    uint16_t FromSocket{};
    bool operator==(const SocketLink &) const = default;
};
enum class HandleParameter : uint8_t { Transform,
                                       ExtrudeOffset,
                                       ExtrudeScale,
                                       BevelWidth,
                                       InsetThickness,
                                       InsetDepth };
// A graph item is the entity its node stands for: an object, bone, scene, material, texture, physics resource, subgraph, or the document.

// A geometry graph's active node and its selected nodes, sorted, on the graph's mesh entity.
// A scene or document graph's nodes are selected through the Selected and Active tags of their entities.
struct GraphActive {
    state::Entity Item{state::Null};
    std::vector<state::Entity> Selection;
    bool Contains(state::Entity node) const { return std::ranges::binary_search(Selection, node); }
};
// Output places the output boundary of a subgraph's page, whose item is the subgraph, and every other cell places the item's one node.
struct GraphCell {
    state::Entity Item;
    numeric::uvec2 Position;
    bool Output{};
};
// The cells the user placed items at on the graph's page, sorted by item, then input boundary before output boundary.
// A subgraph's page shows its interface as an input boundary and an output boundary, both standing for the subgraph.
// The editor lays out every other item from the page's connections.
struct GraphLayout {
    std::vector<GraphCell> Cells;
};
struct PromotedField {
    state::Entity Entity{state::Null};
    state::TypeKey Component{};
    uint16_t Offset{};
    auto operator<=>(const PromotedField &) const = default;
};
enum class ReferenceKind : uint8_t {
    Parent,
    Geometry,
    Skin,
    Constraint,
    ColliderMesh,
    PhysicsMaterial,
    CollisionFilter,
    JointBody,
    JointDescription,
    TriggerFilter,
    TriggerMember,
    Material,
    Texture,
    InstanceSubgraph,
    FilterSystem,
    FilterCollideSystem
};
struct SceneReference {
    state::Entity Owner{state::Null};
    ReferenceKind Kind{};
    uint32_t Index{};
    auto operator<=>(const SceneReference &) const = default;
};
using PortBinding = std::variant<SocketLink, PromotedField, SceneReference, state::Entity>;
// The first object, in entity order, that draws each geometry, sorted by geometry.
using GeometryDrawers = std::vector<std::pair<state::Entity, state::Entity>>;
GeometryDrawers DrawersOf(const state::Registry &);
// The item the reference holds, or none.
// A geometry reference holds the first other object that draws the same geometry.
// A list reference has one reference per entry and one more, past the last entry, that adds an entry.
state::Entity ReferenceValue(const state::Registry &, SceneReference);
std::string ReferenceLabel(const state::Registry &, SceneReference);
// Every reference accepts no item, which clears it.
// A cleared geometry reference gives the object its own copy, a cleared collider geometry uses the object's own geometry, a cleared material slot holds the default material,
// a cleared instance subgraph makes the instance unique, and a cleared list entry is removed.
// Any other cleared reference is empty.
bool ReferenceAccepts(const state::Registry &, SceneReference, state::Entity);
std::vector<state::Entity> ReferenceCandidates(const state::Registry &, SceneReference);
// The references that hold the item.
std::vector<SceneReference> ReferencesTo(const state::Registry &, state::Entity);
std::string ItemName(const state::Registry &, state::Entity);

// A reference and the item it holds.
struct ItemReference {
    ReferenceKind Kind{};
    uint32_t Index{};
    state::Entity Value{state::Null};
    bool operator==(const ItemReference &) const = default;
};
// Derived on each item entity: what the item's node shows from its own entity.
// An object's node holds its component references, with the geometry a geometry or collider geometry reference names.
// A geometry's node holds its material slots, which every object that draws it shows.
struct ItemNode {
    std::string Title;
    std::vector<ItemReference> References;
    bool operator==(const ItemNode &) const = default;
};
// Fills `shown` with the references the item's node shows: an entity's stored references, then the material slots of the geometry it draws, or a material's texture slots.
void ShownReferences(const state::Registry &, state::Entity, const GeometryDrawers &, std::vector<ItemReference> &shown);
// Whether each entity that has a node stores the node its components derive, and no other entity stores one.
bool NodesSettled(const state::Registry &);
// Nodes lists the entities whose node may differ, Pages those whose change alters which items a page shows, Cells the graphs whose placed cells changed,
// and Fields the entities that lost a component a value port can address.
// Revision advances in a pass where a node, a graph node's name or operation, a page's items or a material buffer changed, CellsRevision where a placed cell changed,
// and GraphRevision where the graph of a geometry graph or a user subgraph changed.
struct GraphChanges {
    state::DirtySet Nodes, Pages, Cells, Fields;
    uint64_t Revision{}, CellsRevision{}, GraphRevision{};
};
// Stores each changed entity's node, advances the revisions, and drops the cells, subgraph items and value ports of entities and fields that no longer exist.
void SettleItemGraphs(state::Registry &, bool restoring);
// Drops each value port, of the scene subgraphs under the root, whose item is no longer inside its subgraph.
void DropUnreachablePorts(state::Registry &, state::Entity root);
// Persistent on a subgraph of a scene's or the document's graph: what it contains.
// ParentEntity is the graph or subgraph whose page shows it, and Items its items and subgraphs, sorted.
// Its items stay items of their scene or of the document, and containment is independent of scene membership and transform parenting.
// A scene subgraph that an instance or a prototype uses stays in the document, with the document as its parent, once it is deleted from its scene.
struct SubgraphItems {
    state::Entity ParentEntity{state::Null};
    std::vector<state::Entity> Items;
};
state::Entity GraphRoot(const state::Registry &, state::Entity graph);
std::vector<state::Entity> GraphItems(const state::Registry &, state::Entity graph);
// Whether the graph's page shows the item: one of GraphItems, or in a geometry graph one of its nodes or, in a subgraph, the subgraph itself.
bool ShowsItem(const state::Registry &, state::Entity graph, state::Entity);
// The immediate child representing an item, or the item itself if it is outside this graph.
state::Entity GraphRepresentative(const state::Registry &, state::Entity graph, state::Entity item);
// The cell the user placed the item at on the graph's page, or the subgraph's output boundary at for `output`, or none.
std::optional<numeric::uvec2> GraphPosition(const state::Registry &, state::Entity graph, state::Entity, bool output = false);
void SetGraphPosition(state::Registry &, state::Entity graph, state::Entity, numeric::uvec2, bool output = false);
void ClearGraphPosition(state::Registry &, state::Entity graph, state::Entity, bool output = false);
void PlaceInGraph(state::Registry &, state::Entity graph, state::Entity object);
// A new subgraph that the graph's page shows, with the items, the interface and a unique name from the prefix.
state::Entity CreateSubgraphOfItems(state::Registry &, state::Entity graph, std::vector<state::Entity> items, SubgraphInterface, std::string_view name);
// Destroys each subgraph of the root's graph, leaving its items.
void RemoveSubgraphs(state::Registry &, state::Entity root);
} // namespace graph
