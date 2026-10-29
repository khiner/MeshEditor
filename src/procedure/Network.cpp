#include "procedure/Network.h"
#include "SortUnique.h"
#include "action/Core.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "animation/Clips.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include "timeline/Timeline.h"

namespace procedure {
state::Entity NetworkRoot(const state::Registry &r, state::Entity network) {
    while (const auto *subnet = r.try_get<const NetworkSubnet>(network)) network = subnet->Parent;
    return network;
}
namespace {
std::vector<NetworkItem> CanonicalItems(const state::Registry &r, state::Entity network) {
    std::vector<NetworkItem> items;
    if (network == state::Document) {
        items.push_back({state::Document});
        const auto append = [&]<typename T>() { for (const auto e : r.view<const T>()) items.push_back({e}); };
        append.template operator()<Scene>();
        append.template operator()<HierarchyDefinition>();
        for (const auto e : r.view<const GeometryProcedure>())
            if (!r.all_of<HierarchyGeometry>(e)) items.push_back({e});
        append.template operator()<Armature>();
        append.template operator()<PhysicsMaterial>();
        append.template operator()<CollisionSystem>();
        append.template operator()<CollisionFilter>();
        append.template operator()<PhysicsJointDef>();
        for (uint32_t i = 0; i < r.get<const Animations>(state::Document).Names.size(); ++i) items.push_back({state::Document, i, ItemKind::Animation});
        for (uint32_t i = 0; i < r.Resources.get<const MaterialStore>().Names.size(); ++i) items.push_back({state::Document, i, ItemKind::Material});
        if (const auto *textures = r.try_get<const MaterializedTextures>(state::Document)) {
            for (uint32_t i = 0; i < textures->Items.size(); ++i) items.push_back({state::Document, i, ItemKind::Texture});
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
        std::vector<state::Entity> generated;
        for (const auto [_, instances] : r.view<const ProcedureInstances>().each()) generated.append_range(instances.Entities);
        SortUnique(generated);
        for (const auto e : r.view<const ObjectKind>()) {
            if (!InScene(r, e, network) || r.any_of<SubElementOf, BoneSubPartOf, BoneJoint>(e)) continue;
            if (std::ranges::binary_search(generated, e)) continue;
            if (const auto *member = r.try_get<const HierarchyMember>(e); member && member->Root != e) continue;
            items.push_back({e});
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
uvec2 Position(const state::Registry &r, state::Entity network, NetworkItem item, std::span<const NetworkItem> items) {
    if (const auto position = NetworkPosition(r, network, item)) return *position;
    const auto index = uint32_t(std::ranges::lower_bound(items, item) - items.begin());
    return {index % 4u * 10u + 1u, index / 4u * 5u + 1u};
}
void EraseItems(state::Registry &r, state::Entity network, std::span<const NetworkItem> items) {
    if (r.all_of<NetworkSubnet>(network)) r.patch<NetworkSubnet>(network, [&](auto &subnet) { std::erase_if(subnet.Members, [&](auto item) { return std::ranges::contains(items, item); }); });
    if (r.all_of<NetworkLayout>(network)) r.patch<NetworkLayout>(network, [&](auto &layout) { std::erase_if(layout.Cells, [&](auto cell) { return std::ranges::contains(items, cell.Item); }); });
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
    std::vector<NetworkCell> cells;
    uvec2 origin{UINT32_MAX, UINT32_MAX};
    for (const auto item : items) {
        const auto position = Position(r, a.Network, item, visible);
        origin = {std::min(origin.x, position.x), std::min(origin.y, position.y)};
        cells.push_back({item, position});
    }
    const auto entity = r.create();
    r.emplace<NetworkSubnet>(entity, a.Network, items);
    r.emplace<SubnetInterface>(entity);
    EmplaceUniqueName(r, entity, "Subnet");
    for (auto &cell : cells) cell.Position = {cell.Position.x - origin.x + 1u, cell.Position.y - origin.y + 1u};
    r.emplace<NetworkLayout>(entity, std::move(cells));
    EraseItems(r, a.Network, items);
    InsertItems(r, entity, items);
    InsertItems(r, a.Network, std::array{NetworkItem{entity}});
    SetNetworkPosition(r, a.Network, {entity}, origin);
    ExposeNetworkReferences(r, entity);
    r.emplace_or_replace<NetworkActive>(a.Network, NetworkItem{entity}, std::vector<NetworkItem>{{entity}});
}
void Apply(state::Registry &r, const UngroupItems &a) {
    const auto *group = r.try_get<const NetworkSubnet>(a.Subnet);
    if (!group) return Fail(r, "The subnet does not exist");
    const auto parent = group->Parent;
    const auto items = NetworkItems(r, a.Subnet);
    const auto origin = Position(r, parent, {a.Subnet}, NetworkItems(r, parent));
    for (const auto item : items) {
        const auto cell = Position(r, a.Subnet, item, items);
        SetNetworkPosition(r, parent, item, {origin.x + cell.x - std::min(cell.x, 1u), origin.y + cell.y - std::min(cell.y, 1u)});
    }
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
    for (const auto item : items) SetNetworkPosition(r, a.Destination, item, Position(r, a.Network, item, visible));
    EraseItems(r, a.Network, items);
    InsertItems(r, a.Destination, items);
    ReconcileNetworkPorts(r);
    for (const auto subnet : SortedEntities(r.view<const NetworkSubnet>())) {
        if (NetworkRoot(r, subnet) == NetworkRoot(r, a.Destination)) ExposeNetworkReferences(r, subnet);
    }
    r.emplace_or_replace<NetworkActive>(a.Destination, items.empty() ? NetworkItem{} : items.back(), items);
    r.emplace_or_replace<NetworkActive>(a.Network);
}
void Apply(state::Registry &r, const RenameNetwork &a) {
    if (!r.any_of<NetworkSubnet, Subnet>(a.Network)) return Fail(r, "The subnet does not exist");
    r.remove<Name>(a.Network);
    EmplaceUniqueName(r, a.Network, a.Name);
}
} // namespace action::nodes
