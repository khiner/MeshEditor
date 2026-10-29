#pragma once

#include "numeric/uvec2.h"
#include "procedure/ProcedureSchema.h"
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
enum class ItemKind : uint8_t { Entity,
                                Material,
                                Texture,
                                Animation };
struct NetworkItem {
    state::Entity Entity{state::Null};
    uint32_t Index{};
    ItemKind Kind{ItemKind::Entity};
    auto operator<=>(const NetworkItem &) const = default;
};
struct NetworkActive {
    NetworkItem Item;
    std::vector<NetworkItem> Selection;
    bool Contains(NetworkItem item) const { return std::ranges::binary_search(Selection, item); }
};
struct NetworkCell {
    NetworkItem Item;
    numeric::uvec2 Position;
};
// Persistent presentation on the network owner; resources keep their existing store identities.
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
    InstanceSource
};
struct SceneReference {
    NetworkItem Owner;
    ReferenceKind Kind{};
    uint32_t Index{};
    auto operator<=>(const SceneReference &) const = default;
};
using PortBinding = std::variant<SocketLink, PromotedField, SceneReference, NetworkItem>;
std::vector<SceneReference> SceneReferences(const state::Registry &, NetworkItem);
NetworkItem ReferenceValue(const state::Registry &, SceneReference);
std::string ReferenceLabel(SceneReference);
bool ReferenceAccepts(const state::Registry &, SceneReference, NetworkItem);
std::vector<NetworkItem> ReferenceCandidates(const state::Registry &, SceneReference);
void ExposeNetworkReferences(state::Registry &, state::Entity subnet);
void ReconcileNetworkPorts(state::Registry &);
// Containment belongs to a network occurrence, independently of scene membership and transform parenting.
struct NetworkSubnet {
    state::Entity Parent{state::Null};
    std::vector<NetworkItem> Members;
};
state::Entity NetworkRoot(const state::Registry &, state::Entity network);
std::vector<NetworkItem> NetworkLeaves(const state::Registry &, state::Entity network);
std::vector<NetworkItem> NetworkItems(const state::Registry &, state::Entity network);
// The immediate child representing an item, or the item itself if it is outside this network.
NetworkItem NetworkRepresentative(const state::Registry &, state::Entity network, NetworkItem item);
std::optional<numeric::uvec2> NetworkPosition(const state::Registry &, state::Entity network, NetworkItem);
void SetNetworkPosition(state::Registry &, state::Entity network, NetworkItem, numeric::uvec2);
void PlaceInNetwork(state::Registry &, state::Entity network, state::Entity object);
void RemoveNetworkSubnets(state::Registry &, state::Entity root);
} // namespace procedure
