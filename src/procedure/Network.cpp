#include "procedure/Network.h"
#include "SortUnique.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "armature/ArmatureComponents.h"
#include "gltf/SourceAssets.h"
#include "mesh/MeshChanges.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include "viewport/InteractionComponents.h"

namespace procedure {
state::Entity NetworkRoot(const state::Registry &r, state::Entity network) {
    while (const auto *subnet = r.try_get<const NetworkSubnet>(network)) network = subnet->Parent;
    return network;
}
namespace {
// The kinds of entity the document page shows beside the document itself: the scenes, the items their objects share, and the user definitions of subnet nodes.
using DocumentKinds = TypeList<Scene, HierarchyDefinition, Material, MaterializedTexture, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, SubnetDefinition>;
// Whether the document or the scene holds the item.
bool IsLeaf(const state::Registry &r, state::Entity root, state::Entity e) {
    if (root == state::Document) {
        return e == state::Document || (r.valid(e) && []<typename... Ks>(const state::Registry &r, state::Entity e, TypeList<Ks...>) { return r.any_of<Ks...>(e); }(r, e, DocumentKinds{}));
    }
    if (!r.valid(e) || !r.all_of<ObjectKind>(e)) return false;
    if (!InScene(r, e, root) || r.any_of<SubElementOf, BoneSubPartOf, BoneJoint>(e)) return false;
    if (const auto *member = r.try_get<const HierarchyMember>(e); member && member->Root != e) return false;
    // An instance a procedure generates shows on its parent object's page.
    const auto parent = ParentOrNull(r, e);
    const auto *object = parent == state::Null ? nullptr : r.try_get<const Instance>(parent);
    const auto *generated = object ? r.try_get<const ProcedureInstances>(object->Entity) : nullptr;
    return !generated || !std::ranges::contains(generated->Entities, e);
}
std::vector<state::Entity> CanonicalItems(const state::Registry &r, state::Entity network) {
    std::vector<state::Entity> items;
    if (network == state::Document) {
        items.push_back(state::Document);
        const auto append = [&]<typename T>() { for (const auto e : r.view<const T>()) items.push_back(e); };
        [&]<typename... Ks>(TypeList<Ks...>) { (append.template operator()<Ks>(), ...); }(DocumentKinds{});
    } else if (const auto *object = r.try_get<const Instance>(network); object && r.all_of<ProcedureInstances>(object->Entity)) {
        for (const auto e : r.get<const ProcedureInstances>(object->Entity).Entities)
            if (ParentOrNull(r, e) == network) items.push_back(e);
    } else if (r.any_of<HierarchyInstance, HierarchyDefinition>(network)) {
        for (const auto e : HierarchyClosure(r, network)) {
            if (e != network && r.any_of<ObjectKind, BoneIndex>(e)) items.push_back(e);
        }
    } else if (const auto *armature = r.try_get<const ArmatureObject>(network)) {
        for (const auto e : armature->BoneEntities) items.push_back(e);
    } else if (r.all_of<Scene>(network)) {
        for (const auto e : r.view<const ObjectKind>()) {
            if (IsLeaf(r, network, e)) items.push_back(e);
        }
    }
    SortUnique(items);
    return items;
}
} // namespace
std::vector<state::Entity> NetworkLeaves(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    auto leaves = CanonicalItems(r, root);
    if (root == network) return leaves;
    std::vector<state::Entity> contained;
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
std::vector<state::Entity> NetworkItems(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    auto items = CanonicalItems(r, root);
    if (const auto *subnet = r.try_get<const NetworkSubnet>(network)) {
        std::erase_if(items, [&](auto item) { return !std::ranges::binary_search(subnet->Members, item); });
    }
    std::vector<state::Entity> contained;
    for (const auto [entity, subnet] : r.view<const NetworkSubnet>().each()) {
        if (subnet.Parent == network) items.push_back(entity);
        if (network == root ? NetworkRoot(r, entity) == root : subnet.Parent == network) contained.append_range(subnet.Members);
    }
    SortUnique(contained);
    std::erase_if(items, [&](auto item) { return std::ranges::binary_search(contained, item); });
    SortUnique(items);
    return items;
}
bool ShowsItem(const state::Registry &r, state::Entity network, state::Entity item) {
    if (r.all_of<NetworkNodes>(network)) {
        return r.valid(item) && (item == network ? r.all_of<SubnetDefinition>(network) : r.all_of<ProcedureNode>(item) && NetworkOf(r, item) == network);
    }
    const auto root = NetworkRoot(r, network);
    // The document and a scene hold too many items to list for one check, so they answer per item.
    if (root != state::Document && !r.all_of<Scene>(root)) return std::ranges::binary_search(NetworkItems(r, network), item);
    if (r.valid(item) && r.all_of<NetworkSubnet>(item)) return r.get<const NetworkSubnet>(item).Parent == network;
    if (!IsLeaf(r, root, item)) return false;
    if (network != root) return std::ranges::binary_search(r.get<const NetworkSubnet>(network).Members, item);
    for (const auto [subnet, members] : r.view<const NetworkSubnet>().each()) {
        if (NetworkRoot(r, subnet) == root && std::ranges::binary_search(members.Members, item)) return false;
    }
    return true;
}
state::Entity NetworkRepresentative(const state::Registry &r, state::Entity network, state::Entity item) {
    const auto root = NetworkRoot(r, network);
    for (const auto [entity, subnet] : r.view<const NetworkSubnet>().each()) {
        if (!std::ranges::binary_search(subnet.Members, item) || NetworkRoot(r, entity) != root) continue;
        for (auto group = entity; const auto *parent = r.try_get<const NetworkSubnet>(group); group = parent->Parent) {
            if (group == network) return item;
            if (parent->Parent == network) return group;
        }
    }
    return item;
}
namespace {
auto CellKey(const NetworkCell &cell) { return std::pair{cell.Item, cell.Output}; }
} // namespace
std::optional<numeric::uvec2> NetworkPosition(const state::Registry &r, state::Entity network, state::Entity item, bool output) {
    if (const auto *layout = r.try_get<const NetworkLayout>(network)) {
        const auto at = std::ranges::lower_bound(layout->Cells, std::pair{item, output}, {}, CellKey);
        if (at != layout->Cells.end() && CellKey(*at) == std::pair{item, output}) return at->Position;
    }
    return {};
}
void SetNetworkPosition(state::Registry &r, state::Entity network, state::Entity item, uvec2 position, bool output) {
    r.get_or_emplace<NetworkLayout>(network);
    r.patch<NetworkLayout>(network, [&](auto &layout) {
        const auto at = std::ranges::lower_bound(layout.Cells, std::pair{item, output}, {}, CellKey);
        if (at != layout.Cells.end() && CellKey(*at) == std::pair{item, output}) at->Position = position;
        else layout.Cells.insert(at, {item, position, output});
    });
}
void ClearNetworkPosition(state::Registry &r, state::Entity network, state::Entity item, bool output) {
    if (!NetworkPosition(r, network, item, output)) return;
    if (r.get<const NetworkLayout>(network).Cells.size() == 1u) r.remove<NetworkLayout>(network);
    else r.patch<NetworkLayout>(network, [&](auto &layout) { layout.Cells.erase(std::ranges::lower_bound(layout.Cells, std::pair{item, output}, {}, CellKey)); });
}
void PlaceInNetwork(state::Registry &r, state::Entity network, state::Entity object) {
    if (!r.all_of<NetworkSubnet>(network) || !InScene(r, object, NetworkRoot(r, network))) return;
    r.patch<NetworkSubnet>(network, [&](auto &subnet) {
        const auto at = std::ranges::lower_bound(subnet.Members, object);
        if (at == subnet.Members.end() || *at != object) subnet.Members.insert(at, object);
    });
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
void EraseItems(state::Registry &r, state::Entity network, std::span<const state::Entity> items) {
    if (r.all_of<NetworkSubnet>(network)) r.patch<NetworkSubnet>(network, [&](auto &subnet) { std::erase_if(subnet.Members, [&](auto item) { return std::ranges::contains(items, item); }); });
    for (const auto item : items) ClearNetworkPosition(r, network, item);
}
// Replaces the selection with the items and makes the last one active.
void SelectNodes(state::Registry &r, std::span<const state::Entity> items) {
    selection::Apply(r, selection::SelectNodes(r, {items.begin(), items.end()}, items.empty() ? state::Null : items.back()));
}
void InsertItems(state::Registry &r, state::Entity network, std::span<const state::Entity> items) {
    if (r.all_of<NetworkSubnet>(network)) r.patch<NetworkSubnet>(network, [&](auto &subnet) { subnet.Members.append_range(items); SortUnique(subnet.Members); });
    for (const auto item : items) {
        if (r.all_of<NetworkSubnet>(item)) r.patch<NetworkSubnet>(item, [&](auto &subnet) { subnet.Parent = network; });
    }
}
// Points every mesh slot that holds the material record at the default record.
// Every other reference to the material holds its entity, which reads as no material once the entity is gone.
void ReleaseMaterial(state::Registry &r, uint32_t index) {
    auto &meshes = r.Resources.get<MeshStore>();
    for (const auto mesh : SortedEntities(r.view<const GeometryProcedure>())) {
        const auto id = DrawnStoreId(r, mesh);
        if (!id || !std::ranges::contains(meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(*id).PrimitiveMaterials), index)) continue;
        std::ranges::replace(meshes.EditPrimitiveMaterials(*id), index, 0u);
        // The event pass refreshes what derives from the mesh's materials.
        r.Resources.get<MeshChanges>().Material.emplace(mesh);
    }
}
// Drops the source texture a deleted texture node materialized, and its image when no other source texture reads it, so an export leaves them out.
// Every later index moves down by one wherever the document holds it.
void RemoveSourceTexture(state::Registry &r, uint32_t texture) {
    const auto *sources = r.try_get<const gltf::SourceAssets>(state::Document);
    if (!sources || texture >= sources->Textures.size()) return;
    static constexpr std::array Images{&gltf::Texture::ImageIndex, &gltf::Texture::WebpImageIndex, &gltf::Texture::BasisuImageIndex, &gltf::Texture::DdsImageIndex};
    auto image = gltf::ResolveImageIndex(sources->Textures[texture]);
    for (uint32_t other = 0; image && other < sources->Textures.size(); ++other) {
        if (other != texture && std::ranges::any_of(Images, [&](auto member) { return sources->Textures[other].*member == image; })) image.reset();
    }
    const auto lower = [](uint32_t removed, uint32_t &index) {
        if (index != InvalidSlot && index > removed) --index;
    };
    r.patch<gltf::SourceAssets>(state::Document, [&](auto &edited) {
        edited.Textures.erase(edited.Textures.begin() + texture);
        if (!image) return;
        edited.Images.erase(edited.Images.begin() + *image);
        for (auto &t : edited.Textures) {
            for (const auto member : Images) {
                if (t.*member) lower(*image, *(t.*member));
            }
        }
    });
    for (const auto e : SortedEntities(r.view<const MaterializedTexture>())) {
        r.patch<MaterializedTexture>(e, [&](auto &t) {
            lower(texture, t.SourceTextureIndex);
            if (image) lower(*image, t.SourceImageIndex);
        });
    }
    for (const auto e : SortedEntities(r.view<const gltf::MaterialSourceMeta>())) {
        r.patch<gltf::MaterialSourceMeta>(e, [&](auto &meta) { for (auto &slot : meta.TextureSlots) lower(texture, slot); });
    }
    if (!image) return;
    // The GPU textures and queued uploads identify their image by the same index.
    auto &store = r.Resources.get<TextureStore>();
    for (auto &entry : store.Textures) lower(*image, entry.SourceImageIndex);
    for (auto &upload : store.PendingUploads) lower(*image, upload.ImageIndex);
}
} // namespace
void Apply(state::Registry &r, const DeleteItems &a) {
    auto items = a.Items;
    // A subnet goes with every item inside it.
    for (size_t i = 0; i < items.size(); ++i) {
        if (r.valid(items[i]) && r.all_of<NetworkSubnet>(items[i])) items.append_range(NetworkItems(r, items[i]));
    }
    SortUnique(items);
    // Objects stay in Pose mode, as they do for the viewport's delete.
    const bool pose = r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose;
    std::vector<state::Entity> objects, destroyed;
    for (const auto e : items) {
        if (e == state::Document || !r.valid(e) || r.any_of<BoneIndex, HierarchyDefinition>(e) || (pose && r.all_of<ObjectKind>(e))) continue;
        if (r.all_of<SubnetDefinition>(e)) {
            if (!DefinitionUsed(r, e)) Apply(r, DeleteDefinition{e});
        } else if (r.all_of<Scene>(e)) {
            scene::Apply(r, scene::Remove{e});
        } else if (r.all_of<ObjectKind>(e)) {
            if (!r.all_of<SubElementOf>(e)) objects.push_back(HierarchyEditTarget(r, e, true));
        } else {
            if (const auto *material = r.try_get<const Material>(e)) ReleaseMaterial(r, material->Index);
            if (r.all_of<MaterialGraph>(e)) destroyed.append_range(NodesOf(r, e));
            if (const auto *texture = r.try_get<const MaterializedTexture>(e)) {
                for (const auto &user : ReferencesTo(r, e)) Apply(r, AssignReference{user, state::Null});
                // An op's image input that read the texture reads none.
                for (const auto op : SortedEntities(r.view<const Op>())) {
                    if (std::ranges::contains(r.get<const Op>(op).Images | std::views::values, e)) r.patch<Op>(op, [&](Op &edited) { std::erase_if(edited.Images, [&](const auto &image) { return image.second == e; }); });
                }
                // The GPU texture goes now, so the sampler slots in use always match the document's textures.
                ReleaseTexture(r, texture->SamplerSlot);
                // A texture node per color space can share one source texture, which goes with the last of them.
                const auto source = texture->SourceTextureIndex;
                const auto shares = [&](state::Entity other) { return other != e && r.get<const MaterializedTexture>(other).SourceTextureIndex == source; };
                if (std::ranges::none_of(r.view<const MaterializedTexture>(), shares)) RemoveSourceTexture(r, source);
            }
            destroyed.push_back(e);
        }
    }
    SortUnique(objects);
    Destroy(r, objects);
    r.destroy(destroyed);
}
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
    InsertItems(r, a.Network, std::array{entity});
    DropUnreachablePorts(r, NetworkRoot(r, a.Network));
    SelectNodes(r, std::array{entity});
}
void Apply(state::Registry &r, const UngroupItems &a) {
    const auto *group = r.try_get<const NetworkSubnet>(a.Subnet);
    if (!group) return Fail(r, "The subnet does not exist");
    const auto parent = group->Parent;
    const auto items = NetworkItems(r, a.Subnet);
    EraseItems(r, parent, std::array{a.Subnet});
    InsertItems(r, parent, items);
    SelectNodes(r, items);
    r.destroy(a.Subnet);
}
void Apply(state::Registry &r, const MoveItems &a) {
    auto items = a.Items;
    SortUnique(items);
    if (a.Network == a.Destination) return;
    const auto visible = NetworkItems(r, a.Network);
    if (!ValidNetwork(r, a.Network) || !ValidNetwork(r, a.Destination) || NetworkRoot(r, a.Network) != NetworkRoot(r, a.Destination) || !std::ranges::includes(visible, items)) return Fail(r, "Move items within the same scene or resource network");
    for (auto parent = a.Destination; parent != state::Null;) {
        if (std::ranges::contains(items, parent)) return Fail(r, "A subnet cannot contain itself");
        const auto *subnet = r.try_get<const NetworkSubnet>(parent);
        parent = subnet ? subnet->Parent : state::Null;
    }
    EraseItems(r, a.Network, items);
    InsertItems(r, a.Destination, items);
    DropUnreachablePorts(r, NetworkRoot(r, a.Destination));
    SelectNodes(r, items);
}
void Apply(state::Registry &r, const RenameItem &a) {
    auto e = a.Item;
    if (!r.valid(e)) return Fail(r, "The item does not exist");
    if (r.all_of<MaterializedTexture>(e)) {
        r.patch<MaterializedTexture>(e, [&](auto &texture) { texture.Params.Name = a.Name; });
        return;
    }
    const auto renamed = [&]<typename T> {
        if (!r.all_of<T>(e)) return false;
        r.patch<T>(e, [&](auto &value) { value.Name = a.Name; });
        return true;
    };
    if ([&]<typename... Ts>(TypeList<Ts...>) { return (renamed.template operator()<Ts>() || ...); }(TypeList<Material, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef>{})) return;
    // Every other item has a unique name, and a hierarchy member takes its from the object it was made from.
    // A subnet node takes its name from its user definition.
    if (const auto *subnet = r.try_get<const Subnet>(e)) {
        if (subnet->DefinitionEntity == state::Null) return Fail(r, "A built-in definition's name does not change");
        e = subnet->DefinitionEntity;
    }
    if (r.all_of<ObjectKind>(e) && !r.any_of<SubElementOf, BoneIndex>(e)) e = HierarchyEditTarget(r, e, true);
    else if (!r.any_of<Scene, NetworkSubnet, SubnetDefinition>(e)) return Fail(r, "The item has no name to change");
    r.remove<Name>(e);
    EmplaceUniqueName(r, e, a.Name);
}
} // namespace action::nodes
