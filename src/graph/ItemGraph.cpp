#include "graph/ItemGraph.h"
#include "SortUnique.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "armature/ArmatureComponents.h"
#include "gltf/SourceAssets.h"
#include "graph/Graph.h"
#include "mesh/MeshChanges.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBufferOps.h"
#include "render/MaterialComponents.h"
#include "render/ObjectGeometry.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/Instance.h"
#include "scene/SceneGraph.h"
#include "viewport/InteractionComponents.h"

namespace graph {
state::Entity GraphRoot(const state::Registry &r, state::Entity graph) {
    while (const auto *subgraph = r.try_get<const SubgraphItems>(graph)) graph = subgraph->ParentEntity;
    return graph;
}
namespace {
// The kinds of entity the document page shows beside the document itself: the scenes, the items their objects share, and the subgraphs that subgraph nodes use.
// A scene subgraph the document keeps for its instances shows there too, with the objects it holds.
using DocumentKinds = TypeList<Scene, Material, MaterializedTexture, PhysicsMaterial, CollisionSystem, CollisionFilter, JointDescription>;
// Whether the document or the scene holds the item.
bool IsLeaf(const state::Registry &r, state::Entity root, state::Entity e) {
    if (root == state::Document && (e == state::Document || (r.valid(e) && ([]<typename... Ks>(const state::Registry &r, state::Entity e, TypeList<Ks...>) { return r.any_of<Ks...>(e); }(r, e, DocumentKinds{}) || r.all_of<Subgraph, GraphNodes>(e))))) return true;
    if (!r.valid(e) || !r.all_of<ObjectKind>(e) || r.any_of<SubElementOf, BoneSubPartOf, BoneJoint, CopyOf>(e)) return false;
    // An object in no scene is an item of a subgraph the document keeps.
    if (root == state::Document) return !RootMembership(r, e) && SubgraphHolding(r, e) != state::Null;
    if (!InScene(r, e, root)) return false;
    // An instance a graph generates shows on its parent object's page.
    const auto parent = ParentOrNull(r, e);
    const auto *object = parent == state::Null ? nullptr : r.try_get<const ObjectGeometry>(parent);
    const auto *generated = object ? r.try_get<const GraphInstances>(object->Entity) : nullptr;
    return !generated || !std::ranges::contains(generated->Entities, e);
}
std::vector<state::Entity> CanonicalItems(const state::Registry &r, state::Entity graph) {
    std::vector<state::Entity> items;
    if (graph == state::Document) {
        items.push_back(state::Document);
        const auto append = [&]<typename T>() { for (const auto e : r.view<const T>()) items.push_back(e); };
        [&]<typename... Ks>(TypeList<Ks...>) { (append.template operator()<Ks>(), ...); }(DocumentKinds{});
        for (const auto e : r.view<const Subgraph, const GraphNodes>()) items.push_back(e);
        for (const auto [subgraph, held] : r.view<const SubgraphItems>().each())
            if (held.ParentEntity == state::Document) items.append_range(SubgraphObjects(r, subgraph));
    } else if (const auto *object = r.try_get<const ObjectGeometry>(graph); object && r.all_of<GraphInstances>(object->Entity)) {
        for (const auto e : r.get<const GraphInstances>(object->Entity).Entities)
            if (ParentOrNull(r, e) == graph) items.push_back(e);
    } else if (const auto *instance = r.try_get<const Instance>(graph)) {
        for (const auto &pair : instance->Copies)
            if (r.any_of<ObjectKind, BoneIndex>(pair.CopyEntity)) items.push_back(pair.CopyEntity);
    } else if (const auto *armature = r.try_get<const ArmatureObject>(graph)) {
        for (const auto e : armature->BoneEntities) items.push_back(e);
    } else if (r.all_of<Scene>(graph)) {
        for (const auto e : r.view<const ObjectKind>()) {
            if (IsLeaf(r, graph, e)) items.push_back(e);
        }
    }
    SortUnique(items);
    return items;
}
} // namespace
std::vector<state::Entity> GraphItems(const state::Registry &r, state::Entity graph) {
    const auto root = GraphRoot(r, graph);
    auto items = CanonicalItems(r, root);
    if (const auto *subgraph = r.try_get<const SubgraphItems>(graph)) {
        std::erase_if(items, [&](auto item) { return !std::ranges::binary_search(subgraph->Items, item); });
    }
    std::vector<state::Entity> contained;
    for (const auto [entity, subgraph] : r.view<const SubgraphItems>().each()) {
        if (subgraph.ParentEntity == graph) items.push_back(entity);
        if (graph == root ? GraphRoot(r, entity) == root : subgraph.ParentEntity == graph) contained.append_range(subgraph.Items);
    }
    SortUnique(contained);
    std::erase_if(items, [&](auto item) { return std::ranges::binary_search(contained, item); });
    SortUnique(items);
    return items;
}
bool ShowsItem(const state::Registry &r, state::Entity graph, state::Entity item) {
    if (r.all_of<GraphNodes>(graph)) {
        return r.valid(item) && (item == graph ? r.all_of<Subgraph>(graph) : r.all_of<GraphNode>(item) && GraphOf(r, item) == graph);
    }
    const auto root = GraphRoot(r, graph);
    // The document and a scene hold too many items to list for one check, so they answer per item.
    if (root != state::Document && !r.all_of<Scene>(root)) return std::ranges::binary_search(GraphItems(r, graph), item);
    if (r.valid(item) && r.all_of<SubgraphItems>(item)) return r.get<const SubgraphItems>(item).ParentEntity == graph;
    if (!IsLeaf(r, root, item)) return false;
    if (graph != root) return std::ranges::binary_search(r.get<const SubgraphItems>(graph).Items, item);
    for (const auto [subgraph, held] : r.view<const SubgraphItems>().each()) {
        if (GraphRoot(r, subgraph) == root && std::ranges::binary_search(held.Items, item)) return false;
    }
    return true;
}
state::Entity GraphRepresentative(const state::Registry &r, state::Entity graph, state::Entity item) {
    const auto root = GraphRoot(r, graph);
    for (const auto [entity, subgraph] : r.view<const SubgraphItems>().each()) {
        if (!std::ranges::binary_search(subgraph.Items, item) || GraphRoot(r, entity) != root) continue;
        for (auto group = entity; const auto *parent = r.try_get<const SubgraphItems>(group); group = parent->ParentEntity) {
            if (group == graph) return item;
            if (parent->ParentEntity == graph) return group;
        }
    }
    return item;
}
namespace {
auto CellKey(const GraphCell &cell) { return std::pair{cell.Item, cell.Output}; }
} // namespace
std::optional<numeric::uvec2> GraphPosition(const state::Registry &r, state::Entity graph, state::Entity item, bool output) {
    if (const auto *layout = r.try_get<const GraphLayout>(graph)) {
        const auto at = std::ranges::lower_bound(layout->Cells, std::pair{item, output}, {}, CellKey);
        if (at != layout->Cells.end() && CellKey(*at) == std::pair{item, output}) return at->Position;
    }
    return {};
}
void SetGraphPosition(state::Registry &r, state::Entity graph, state::Entity item, uvec2 position, bool output) {
    r.get_or_emplace<GraphLayout>(graph);
    r.patch<GraphLayout>(graph, [&](auto &layout) {
        const auto at = std::ranges::lower_bound(layout.Cells, std::pair{item, output}, {}, CellKey);
        if (at != layout.Cells.end() && CellKey(*at) == std::pair{item, output}) at->Position = position;
        else layout.Cells.insert(at, {item, position, output});
    });
}
void ClearGraphPosition(state::Registry &r, state::Entity graph, state::Entity item, bool output) {
    if (!GraphPosition(r, graph, item, output)) return;
    if (r.get<const GraphLayout>(graph).Cells.size() == 1u) r.remove<GraphLayout>(graph);
    else r.patch<GraphLayout>(graph, [&](auto &layout) { layout.Cells.erase(std::ranges::lower_bound(layout.Cells, std::pair{item, output}, {}, CellKey)); });
}
void PlaceInGraph(state::Registry &r, state::Entity graph, state::Entity object) {
    if (!r.all_of<SubgraphItems>(graph) || !InScene(r, object, GraphRoot(r, graph))) return;
    r.patch<SubgraphItems>(graph, [&](auto &subgraph) {
        const auto at = std::ranges::lower_bound(subgraph.Items, object);
        if (at == subgraph.Items.end() || *at != object) subgraph.Items.insert(at, object);
    });
}
state::Entity CreateSubgraphOfItems(state::Registry &r, state::Entity graph, std::vector<state::Entity> items, SubgraphInterface interface, std::string_view name) {
    const auto subgraph = r.create();
    r.emplace<Subgraph>(subgraph);
    r.emplace<SubgraphItems>(subgraph, graph, std::move(items));
    r.emplace<SubgraphInterface>(subgraph, std::move(interface));
    EmplaceUniqueName(r, subgraph, name);
    if (r.all_of<SubgraphItems>(graph)) r.patch<SubgraphItems>(graph, [&](auto &outer) { outer.Items.insert(std::ranges::lower_bound(outer.Items, subgraph), subgraph); });
    return subgraph;
}
void RemoveSubgraphs(state::Registry &r, state::Entity root) {
    std::vector<state::Entity> removed;
    for (const auto entity : r.view<const SubgraphItems>())
        if (GraphRoot(r, entity) == root) removed.push_back(entity);
    r.destroy(removed);
}
} // namespace graph

namespace action::nodes {
namespace {
using namespace graph;
bool ValidGraph(const state::Registry &r, state::Entity graph) {
    const auto root = GraphRoot(r, graph);
    const auto *object = r.try_get<const ObjectGeometry>(root);
    return root == state::Document || r.any_of<Scene, ArmatureObject, Instance>(root) || (object && r.all_of<GraphInstances>(object->Entity));
}
// An item that leaves a graph's page leaves its cell there.
void EraseItems(state::Registry &r, state::Entity graph, std::span<const state::Entity> items) {
    if (r.all_of<SubgraphItems>(graph)) r.patch<SubgraphItems>(graph, [&](auto &subgraph) { std::erase_if(subgraph.Items, [&](auto item) { return std::ranges::contains(items, item); }); });
    for (const auto item : items) ClearGraphPosition(r, graph, item);
}
// Replaces the selection with the items and makes the last one active.
void SelectNodes(state::Registry &r, std::span<const state::Entity> items) {
    selection::Apply(r, selection::SelectNodes(r, {items.begin(), items.end()}, items.empty() ? state::Null : items.back()));
}
void InsertItems(state::Registry &r, state::Entity graph, std::span<const state::Entity> items) {
    if (r.all_of<SubgraphItems>(graph)) r.patch<SubgraphItems>(graph, [&](auto &subgraph) { subgraph.Items.append_range(items); SortUnique(subgraph.Items); });
    for (const auto item : items) {
        if (r.all_of<SubgraphItems>(item)) r.patch<SubgraphItems>(item, [&](auto &subgraph) { subgraph.ParentEntity = graph; });
    }
}
// Points every mesh slot that holds the material record at the default record.
// Every other reference to the material holds its entity, which reads as no material once the entity is gone.
void ReleaseMaterial(state::Registry &r, uint32_t index) {
    auto &meshes = r.Resources.get<MeshStore>();
    for (const auto mesh : SortedEntities(r.view<const GeometryGraph>())) {
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
    // A subgraph goes with every item inside it, except a scene subgraph that an instance or a prototype uses, which moves to the document with its items.
    for (size_t i = 0; i < items.size(); ++i) {
        if (!r.valid(items[i]) || !r.all_of<SubgraphItems>(items[i])) continue;
        if (IsSceneSubgraph(r, items[i]) && SubgraphInstanced(r, items[i])) {
            if (GraphRoot(r, items[i]) != state::Document) UnplaceSubgraph(r, items[i]);
            items[i] = state::Null;
        } else items.append_range(GraphItems(r, items[i]));
    }
    SortUnique(items);
    // Objects stay in Pose mode, as they do for the viewport's delete.
    const bool pose = r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose;
    std::vector<state::Entity> objects, destroyed;
    for (const auto e : items) {
        if (e == state::Document || !r.valid(e) || r.all_of<BoneIndex>(e) || (pose && r.all_of<ObjectKind>(e))) continue;
        if (r.all_of<Subgraph, GraphNodes>(e)) {
            if (!SubgraphUsed(r, e)) Apply(r, DeleteSubgraph{e});
        } else if (r.all_of<Scene>(e)) {
            scene::Apply(r, scene::Remove{e});
        } else if (r.all_of<ObjectKind>(e)) {
            if (!r.all_of<SubElementOf>(e)) objects.push_back(SourceOfCopy(r, e));
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
void CollapseItems(state::Registry &r, const Collapse &a) {
    auto items = a.Nodes;
    SortUnique(items);
    const auto visible = GraphItems(r, a.Graph);
    if (items.empty() || !ValidGraph(r, a.Graph) || !std::ranges::includes(visible, items)) return Fail(r, "Collapse items of one graph");
    const auto entity = CreateSubgraphOfItems(r, a.Graph, items, {}, "Subgraph");
    EraseItems(r, a.Graph, items);
    InsertItems(r, entity, items);
    DropUnreachablePorts(r, GraphRoot(r, a.Graph));
    SelectNodes(r, std::array{entity});
}
namespace {
// Replaces the subgraph with its items, or the object's Instance with copies of its subgraph's items below the object, and appends what replaced it.
// It reports a failure and returns false for a subgraph an Instance uses, or an object that has no Instance or is not placed in a scene.
bool ExpandItem(state::Registry &r, state::Entity node, bool all_levels, std::vector<state::Entity> &expanded) {
    const bool instance = r.all_of<Instance>(node);
    if (!instance && SubgraphInstanced(r, node)) {
        Fail(r, "Make each instance of the subgraph unique before expanding it");
        return false;
    }
    const auto subgraph = instance ? MakeInstanceUnique(r, node) : node;
    if (subgraph == state::Null) return false;
    if (instance) expanded.push_back(node);
    const auto parent = r.get<const SubgraphItems>(subgraph).ParentEntity;
    const auto items = GraphItems(r, subgraph);
    EraseItems(r, parent, std::array{subgraph});
    InsertItems(r, parent, items);
    r.destroy(subgraph);
    // An object with an Instance among a graph's items is placed in a scene.
    for (const auto item : items) {
        if (all_levels && r.all_of<Instance>(item)) ExpandItem(r, item, true, expanded);
        else expanded.push_back(item);
    }
    return true;
}
} // namespace
void ExpandItems(state::Registry &r, const Expand &a) {
    std::vector<state::Entity> expanded;
    if (ExpandItem(r, a.Node, a.AllLevels, expanded)) SelectNodes(r, expanded);
}
void Apply(state::Registry &r, const MoveItems &a) {
    auto items = a.Items;
    SortUnique(items);
    if (a.Graph == a.Destination) return;
    const auto visible = GraphItems(r, a.Graph);
    if (!ValidGraph(r, a.Graph) || !ValidGraph(r, a.Destination) || GraphRoot(r, a.Graph) != GraphRoot(r, a.Destination) || !std::ranges::includes(visible, items)) return Fail(r, "Move items within the same scene or resource graph");
    for (auto parent = a.Destination; parent != state::Null;) {
        if (std::ranges::contains(items, parent)) return Fail(r, "A subgraph cannot contain itself");
        const auto *subgraph = r.try_get<const SubgraphItems>(parent);
        parent = subgraph ? subgraph->ParentEntity : state::Null;
    }
    for (const auto item : items) {
        const auto objects = r.all_of<SubgraphItems>(item) ? SubgraphObjects(r, item) : std::vector{item};
        for (const auto object : objects) {
            for (const auto used : UsedSubgraphs(r, object))
                if (UseRecurses(r, used, a.Destination)) return Fail(r, "A subgraph cannot hold an instance of itself");
            for (auto above = ParentOrNull(r, object); above != state::Null; above = ParentOrNull(r, above))
                if (r.all_of<Instance>(above) && UseRecurses(r, r.get<const Instance>(above).SubgraphEntity, a.Destination)) return Fail(r, "A subgraph cannot hold an item below an object with an Instance of that subgraph");
        }
    }
    EraseItems(r, a.Graph, items);
    InsertItems(r, a.Destination, items);
    DropUnreachablePorts(r, GraphRoot(r, a.Destination));
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
    if ([&]<typename... Ts>(TypeList<Ts...>) { return (renamed.template operator()<Ts>() || ...); }(TypeList<Material, PhysicsMaterial, CollisionSystem, CollisionFilter, JointDescription>{})) return;
    // Every other item has a unique name, and an instance's copy takes its from the object it copies.
    // A subgraph node takes its name from its user subgraph.
    if (const auto *subgraph = r.all_of<GraphNode>(e) ? r.try_get<const SubgraphNode>(e) : nullptr) {
        if (UserSubgraph(subgraph->SubgraphId) == state::Null) return Fail(r, "A built-in subgraph's name does not change");
        e = UserSubgraph(subgraph->SubgraphId);
    }
    if (r.all_of<ObjectKind>(e) && !r.any_of<SubElementOf, BoneIndex>(e)) e = SourceOfCopy(r, e);
    else if (!r.any_of<Scene, Subgraph>(e)) return Fail(r, "The item has no name to change");
    r.remove<Name>(e);
    EmplaceUniqueName(r, e, a.Name);
}
} // namespace action::nodes
