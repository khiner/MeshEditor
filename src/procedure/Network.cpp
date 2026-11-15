#include "procedure/Network.h"
#include "SortUnique.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "armature/ArmatureComponents.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"

namespace procedure {
state::Entity NetworkRoot(const state::Registry &r, state::Entity network) {
    while (const auto *subnet = r.try_get<const NetworkSubnet>(network)) network = subnet->Parent;
    return network;
}
uint32_t ItemCount(const state::Registry &r, ItemKind kind) {
    if (kind == ItemKind::Material) return uint32_t(r.Resources.get<const MaterialStore>().Names.size());
    const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
    return textures ? uint32_t(textures->Items.size()) : 0u;
}
namespace {
// The kinds of entity the document page shows: the scenes and the items their objects share.
using DocumentKinds = TypeList<Scene, HierarchyDefinition, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef>;
// Whether the document or the scene holds the item.
bool IsLeaf(const state::Registry &r, state::Entity root, NetworkItem item) {
    const auto e = item.Entity;
    if (root == state::Document) {
        if (item.Kind != ItemKind::Entity) return item.Index < ItemCount(r, item.Kind);
        return r.valid(e) && []<typename... Ks>(const state::Registry &r, state::Entity e, TypeList<Ks...>) { return r.any_of<Ks...>(e); }(r, e, DocumentKinds{});
    }
    if (item.Kind != ItemKind::Entity || !r.valid(e) || !r.all_of<ObjectKind>(e)) return false;
    if (!InScene(r, e, root) || r.any_of<SubElementOf, BoneSubPartOf, BoneJoint>(e)) return false;
    if (const auto *member = r.try_get<const HierarchyMember>(e); member && member->Root != e) return false;
    // An instance a procedure generates shows on its parent object's page.
    const auto parent = ParentOrNull(r, e);
    const auto *object = parent == state::Null ? nullptr : r.try_get<const Instance>(parent);
    const auto *generated = object ? r.try_get<const ProcedureInstances>(object->Entity) : nullptr;
    return !generated || !std::ranges::contains(generated->Entities, e);
}
std::vector<NetworkItem> CanonicalItems(const state::Registry &r, state::Entity network) {
    std::vector<NetworkItem> items;
    if (network == state::Document) {
        const auto append = [&]<typename T>() { for (const auto e : r.view<const T>()) items.push_back({e}); };
        [&]<typename... Ks>(TypeList<Ks...>) { (append.template operator()<Ks>(), ...); }(DocumentKinds{});
        for (const auto kind : {ItemKind::Material, ItemKind::Texture}) {
            for (uint32_t i = 0; i < ItemCount(r, kind); ++i) items.push_back({state::Document, i, kind});
        }
    } else if (const auto *object = r.try_get<const Instance>(network); object && r.all_of<ProcedureInstances>(object->Entity)) {
        for (const auto e : r.get<const ProcedureInstances>(object->Entity).Entities)
            if (ParentOrNull(r, e) == network) items.push_back({e});
    } else if (r.any_of<HierarchyInstance, HierarchyDefinition>(network)) {
        for (const auto e : HierarchyClosure(r, network)) {
            if (e != network && r.any_of<ObjectKind, BoneIndex>(e)) items.push_back({e});
        }
    } else if (const auto *armature = r.try_get<const ArmatureObject>(network)) {
        for (const auto e : armature->BoneEntities) items.push_back({e});
    } else if (r.all_of<Scene>(network)) {
        for (const auto e : r.view<const ObjectKind>()) {
            if (IsLeaf(r, network, {e})) items.push_back({e});
        }
    }
    SortUnique(items);
    return items;
}
} // namespace
std::vector<NetworkItem> NetworkLeaves(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    auto leaves = CanonicalItems(r, root);
    if (root == network) return leaves;
    std::vector<NetworkItem> contained;
    for (const auto [entity, subnet] : r.view<const NetworkSubnet>().each()) {
        for (auto parent = entity; const auto *group = r.try_get<const NetworkSubnet>(parent); parent = group->Parent) {
            if (parent != network) continue;
            contained.append_range(subnet.Members);
            break;
        }
    }
    SortUnique(contained);
    std::erase_if(leaves, [&](auto item) { return !std::ranges::binary_search(contained, item); });
    return leaves;
}
std::vector<NetworkItem> NetworkItems(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    auto items = CanonicalItems(r, root);
    if (const auto *subnet = r.try_get<const NetworkSubnet>(network)) {
        std::erase_if(items, [&](auto item) { return !std::ranges::binary_search(subnet->Members, item); });
    }
    std::vector<NetworkItem> contained;
    for (const auto [entity, subnet] : r.view<const NetworkSubnet>().each()) {
        if (subnet.Parent == network) items.push_back({entity});
        if (network == root ? NetworkRoot(r, entity) == root : subnet.Parent == network) contained.append_range(subnet.Members);
    }
    SortUnique(contained);
    std::erase_if(items, [&](auto item) { return std::ranges::binary_search(contained, item); });
    SortUnique(items);
    return items;
}
bool ShowsItem(const state::Registry &r, state::Entity network, NetworkItem item) {
    const auto e = item.Entity;
    if (const auto owner = r.all_of<Subnet>(network) ? r.get<const ProcedureNode>(network).Procedure : network; r.all_of<GeometryProcedure>(owner)) {
        return item.Kind == ItemKind::Entity && r.valid(e) && r.all_of<ProcedureNode>(e) && (e == network ? owner != network : NetworkOf(r, e) == network);
    }
    const auto root = NetworkRoot(r, network);
    // The document and a scene hold too many items to list for one check, so they answer per item.
    if (root != state::Document && !r.all_of<Scene>(root)) return std::ranges::binary_search(NetworkItems(r, network), item);
    if (item.Kind == ItemKind::Entity && r.valid(e) && r.all_of<NetworkSubnet>(e)) return r.get<const NetworkSubnet>(e).Parent == network;
    if (!IsLeaf(r, root, item)) return false;
    if (network != root) return std::ranges::binary_search(r.get<const NetworkSubnet>(network).Members, item);
    for (const auto [subnet, members] : r.view<const NetworkSubnet>().each()) {
        if (NetworkRoot(r, subnet) == root && std::ranges::binary_search(members.Members, item)) return false;
    }
    return true;
}
NetworkItem NetworkRepresentative(const state::Registry &r, state::Entity network, NetworkItem item) {
    const auto root = NetworkRoot(r, network);
    for (const auto [entity, subnet] : r.view<const NetworkSubnet>().each()) {
        if (!std::ranges::binary_search(subnet.Members, item) || NetworkRoot(r, entity) != root) continue;
        for (auto group = entity; const auto *parent = r.try_get<const NetworkSubnet>(group); group = parent->Parent) {
            if (group == network) return item;
            if (parent->Parent == network) return {group};
        }
    }
    return item;
}
std::optional<numeric::uvec2> NetworkPosition(const state::Registry &r, state::Entity network, NetworkItem item) {
    if (const auto *layout = r.try_get<const NetworkLayout>(network)) {
        const auto at = std::ranges::lower_bound(layout->Cells, item, {}, &NetworkCell::Item);
        if (at != layout->Cells.end() && at->Item == item) return at->Position;
    }
    return {};
}
void SetNetworkPosition(state::Registry &r, state::Entity network, NetworkItem item, uvec2 position) {
    r.get_or_emplace<NetworkLayout>(network);
    r.patch<NetworkLayout>(network, [&](auto &layout) {
        const auto at = std::ranges::lower_bound(layout.Cells, item, {}, &NetworkCell::Item);
        if (at != layout.Cells.end() && at->Item == item) at->Position = position;
        else layout.Cells.insert(at, {item, position});
    });
}
void ClearNetworkPosition(state::Registry &r, state::Entity network, NetworkItem item) {
    if (!NetworkPosition(r, network, item)) return;
    if (r.get<const NetworkLayout>(network).Cells.size() == 1u) r.remove<NetworkLayout>(network);
    else r.patch<NetworkLayout>(network, [&](auto &layout) { layout.Cells.erase(std::ranges::lower_bound(layout.Cells, item, {}, &NetworkCell::Item)); });
}
void PlaceInNetwork(state::Registry &r, state::Entity network, state::Entity object) {
    if (!r.all_of<NetworkSubnet>(network) || !InScene(r, object, NetworkRoot(r, network))) return;
    r.patch<NetworkSubnet>(network, [&](auto &subnet) {
        const NetworkItem item{object};
        const auto at = std::ranges::lower_bound(subnet.Members, item);
        if (at == subnet.Members.end() || *at != item) subnet.Members.insert(at, item);
    });
    r.emplace_or_replace<NetworkActive>(network, NetworkItem{object}, std::vector<NetworkItem>{{object}});
}
void RemoveNetworkSubnets(state::Registry &r, state::Entity root) {
    std::vector<state::Entity> removed;
    for (const auto entity : r.view<const NetworkSubnet>())
        if (NetworkRoot(r, entity) == root) removed.push_back(entity);
    r.destroy(removed);
}
} // namespace procedure

namespace action::nodes {
namespace {
using namespace procedure;
bool ValidNetwork(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    const auto *object = r.try_get<const Instance>(root);
    return root == state::Document || r.any_of<Scene, ArmatureObject, HierarchyInstance, HierarchyDefinition>(root) || (object && r.all_of<ProcedureInstances>(object->Entity));
}
// An item that leaves a network's page leaves its cell there.
void EraseItems(state::Registry &r, state::Entity network, std::span<const NetworkItem> items) {
    if (r.all_of<NetworkSubnet>(network)) r.patch<NetworkSubnet>(network, [&](auto &subnet) { std::erase_if(subnet.Members, [&](auto item) { return std::ranges::contains(items, item); }); });
    for (const auto item : items) ClearNetworkPosition(r, network, item);
}
void InsertItems(state::Registry &r, state::Entity network, std::span<const NetworkItem> items) {
    if (r.all_of<NetworkSubnet>(network)) r.patch<NetworkSubnet>(network, [&](auto &subnet) { subnet.Members.append_range(items); SortUnique(subnet.Members); });
    for (const auto item : items) {
        if (item.Kind == ItemKind::Entity && r.all_of<NetworkSubnet>(item.Entity)) r.patch<NetworkSubnet>(item.Entity, [&](auto &subnet) { subnet.Parent = network; });
    }
}
} // namespace
void Apply(state::Registry &r, const GroupItems &a) {
    auto items = a.Items;
    SortUnique(items);
    const auto visible = NetworkItems(r, a.Network);
    if (items.empty() || !ValidNetwork(r, a.Network) || !std::ranges::includes(visible, items)) return Fail(r, "Grouping requires items in one network");
    const auto entity = r.create();
    r.emplace<NetworkSubnet>(entity, a.Network, items);
    r.emplace<SubnetInterface>(entity);
    EmplaceUniqueName(r, entity, "Subnet");
    EraseItems(r, a.Network, items);
    InsertItems(r, entity, items);
    InsertItems(r, a.Network, std::array{NetworkItem{entity}});
    DropUnreachablePorts(r, NetworkRoot(r, a.Network));
    r.emplace_or_replace<NetworkActive>(a.Network, NetworkItem{entity}, std::vector<NetworkItem>{{entity}});
}
void Apply(state::Registry &r, const UngroupItems &a) {
    const auto *group = r.try_get<const NetworkSubnet>(a.Subnet);
    if (!group) return Fail(r, "The subnet does not exist");
    const auto parent = group->Parent;
    const auto items = NetworkItems(r, a.Subnet);
    EraseItems(r, parent, std::array{NetworkItem{a.Subnet}});
    InsertItems(r, parent, items);
    r.emplace_or_replace<NetworkActive>(parent, items.empty() ? NetworkItem{} : items.back(), items);
    r.destroy(a.Subnet);
}
void Apply(state::Registry &r, const MoveItems &a) {
    auto items = a.Items;
    SortUnique(items);
    if (a.Network == a.Destination) return;
    const auto visible = NetworkItems(r, a.Network);
    if (!ValidNetwork(r, a.Network) || !ValidNetwork(r, a.Destination) || NetworkRoot(r, a.Network) != NetworkRoot(r, a.Destination) || !std::ranges::includes(visible, items)) return Fail(r, "Move items within the same scene or resource network");
    for (auto parent = a.Destination; parent != state::Null;) {
        if (std::ranges::contains(items, NetworkItem{parent})) return Fail(r, "A subnet cannot contain itself");
        const auto *subnet = r.try_get<const NetworkSubnet>(parent);
        parent = subnet ? subnet->Parent : state::Null;
    }
    EraseItems(r, a.Network, items);
    InsertItems(r, a.Destination, items);
    DropUnreachablePorts(r, NetworkRoot(r, a.Destination));
    r.emplace_or_replace<NetworkActive>(a.Destination, items.empty() ? NetworkItem{} : items.back(), items);
    r.emplace_or_replace<NetworkActive>(a.Network);
}
void Apply(state::Registry &r, const RenameNetwork &a) {
    if (!r.any_of<NetworkSubnet, Subnet>(a.Network)) return Fail(r, "The subnet does not exist");
    r.remove<Name>(a.Network);
    EmplaceUniqueName(r, a.Network, a.Name);
}
} // namespace action::nodes
