#pragma once

#include "numeric/uvec2.h"
#include "procedure/ProcedureSchema.h"
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

namespace procedure {
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
// A network item is the entity its node stands for: an object, bone, scene, material, texture, physics resource, subnet, or the document.

// A geometry procedure's active node and its selected nodes, sorted, on the procedure's mesh entity.
// A scene or document network's nodes are selected through the Selected and Active tags of their entities.
struct NetworkActive {
    state::Entity Item{state::Null};
    std::vector<state::Entity> Selection;
    bool Contains(state::Entity node) const { return std::ranges::binary_search(Selection, node); }
};
// Output places the output boundary of a definition's page, whose item is the definition, and every other cell places the item's one node.
struct NetworkCell {
    state::Entity Item;
    numeric::uvec2 Position;
    bool Output{};
};
// The cells the user placed items at on the network's page, sorted by item, then input boundary before output boundary.
// A definition's page shows its interface as an input boundary and an output boundary, both standing for the definition.
// The editor lays out every other item from the page's connections.
struct NetworkLayout {
    std::vector<NetworkCell> Cells;
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
    JointDefinition,
    TriggerFilter,
    TriggerMember,
    Material,
    Texture,
    InstanceSource,
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
// a cleared instance source makes the instance an independent copy, and a cleared list entry is removed.
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
struct NetworkNode {
    std::string Title;
    std::vector<ItemReference> References;
    bool operator==(const NetworkNode &) const = default;
};
// Fills `shown` with the references the item's node shows: an entity's stored references, then the material slots of the geometry it draws, or a material's texture slots.
void ShownReferences(const state::Registry &, state::Entity, const GeometryDrawers &, std::vector<ItemReference> &shown);
// Whether each entity that has a node stores the node its components derive, and no other entity stores one.
bool NodesSettled(const state::Registry &);
// Nodes lists the entities whose node may differ, Pages those whose change alters which items a page shows, Cells the networks whose placed cells changed,
// and Fields the entities that lost a component a value port can address.
// Revision advances in a pass where a node, a procedure node's name or operation, a page's items or a material buffer changed, CellsRevision where a placed cell changed,
// and GraphRevision where the graph of a geometry procedure or a user definition changed.
struct NetworkChanges {
    state::DirtySet Nodes, Pages, Cells, Fields;
    uint64_t Revision{}, CellsRevision{}, GraphRevision{};
};
// Stores each changed entity's node, advances the revisions, and drops the cells, subnet members and value ports of entities and fields that no longer exist.
void SettleNetworks(state::Registry &, bool restoring);
// Drops each value port, of the scene subnets under the root, whose item is no longer inside its subnet.
void DropUnreachablePorts(state::Registry &, state::Entity root);
// Containment belongs to a network occurrence, independently of scene membership and transform parenting.
struct NetworkSubnet {
    state::Entity Parent{state::Null};
    std::vector<state::Entity> Members;
};
state::Entity NetworkRoot(const state::Registry &, state::Entity network);
std::vector<state::Entity> NetworkLeaves(const state::Registry &, state::Entity network);
std::vector<state::Entity> NetworkItems(const state::Registry &, state::Entity network);
// Whether the network's page shows the item: one of NetworkItems, or in a geometry network one of its nodes or, in a definition's network, the definition itself.
bool ShowsItem(const state::Registry &, state::Entity network, state::Entity);
// The immediate child representing an item, or the item itself if it is outside this network.
state::Entity NetworkRepresentative(const state::Registry &, state::Entity network, state::Entity item);
// The cell the user placed the item at on the network's page, or the definition's output boundary at for `output`, or none.
std::optional<numeric::uvec2> NetworkPosition(const state::Registry &, state::Entity network, state::Entity, bool output = false);
void SetNetworkPosition(state::Registry &, state::Entity network, state::Entity, numeric::uvec2, bool output = false);
void ClearNetworkPosition(state::Registry &, state::Entity network, state::Entity, bool output = false);
void PlaceInNetwork(state::Registry &, state::Entity network, state::Entity object);
void RemoveNetworkSubnets(state::Registry &, state::Entity root);
} // namespace procedure
