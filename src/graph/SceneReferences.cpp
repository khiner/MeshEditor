#include "SortUnique.h"
#include "action/Bone.h"
#include "action/Build.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "armature/BoneConstraint.h"
#include "graph/Graph.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBufferOps.h"
#include "render/MaterialComponents.h"
#include "render/ObjectGeometry.h"
#include "render/Textures.h"

namespace graph {
namespace {
// The geometry the object draws, or null.
state::Entity DrawnGeometry(const state::Registry &r, state::Entity object) {
    const auto *geometry = r.try_get<const ObjectGeometry>(object);
    return geometry && r.all_of<GeometryGraph>(geometry->Entity) ? geometry->Entity : state::Null;
}
state::Entity FirstDrawer(const GeometryDrawers &drawers, state::Entity geometry) {
    const auto at = std::ranges::lower_bound(drawers, geometry, {}, &GeometryDrawers::value_type::first);
    return at != drawers.end() && at->first == geometry ? at->second : state::Null;
}
// The material of each of the geometry's primitives, or none for a null geometry or one without a mesh.
std::span<const uint32_t> MaterialSlots(const state::Registry &r, state::Entity geometry) {
    if (geometry == state::Null || !HasMesh(r, geometry)) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    return meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(GetMesh(r, geometry).GetStoreId()).PrimitiveMaterials);
}
// The list a list reference indexes, or null for any other reference.
const std::vector<state::Entity> *ReferenceList(const state::Registry &r, SceneReference reference) {
    const auto e = reference.Owner;
    if (reference.Kind == ReferenceKind::TriggerMember) return &r.get<const TriggerNodes>(e).Nodes;
    if (reference.Kind == ReferenceKind::FilterSystem) return &r.get<const ::CollisionFilter>(e).Systems;
    if (reference.Kind == ReferenceKind::FilterCollideSystem) return &r.get<const ::CollisionFilter>(e).CollideSystems;
    return nullptr;
}
void ForEachReference(const state::Registry &r, state::Entity e, auto &&visit) {
    const auto add = [&](ReferenceKind kind, uint32_t index = 0u) { visit(SceneReference{e, kind, index}); };
    const auto add_list = [&](ReferenceKind kind, size_t entries) {
        for (uint32_t i = 0u; i <= entries; ++i) add(kind, i);
    };
    if (!r.valid(e)) return;
    if (const auto *record = r.try_get<const ::Material>(e)) {
        const auto &material = GetMaterials(r)[record->Index];
        const auto fields = [&](this const auto &self, const auto &value) -> void {
            using T = std::remove_cvref_t<decltype(value)>;
            if constexpr (std::same_as<T, TextureInfo>) add(ReferenceKind::Texture, uint32_t(reinterpret_cast<const std::byte *>(&value.Slot) - reinterpret_cast<const std::byte *>(&material)));
            else if constexpr (field::Walkable<T>) field::ForEach(value, [&](const auto &member, auto) { self(member); });
        };
        fields(material);
        return;
    }
    if (r.all_of<ObjectKind>(e) && !r.any_of<BoneIndex, SubElementOf>(e)) add(ReferenceKind::Parent);
    const auto geometry = DrawnGeometry(r, e);
    if (geometry != state::Null) add(ReferenceKind::Geometry);
    if (r.all_of<Instance>(e)) add(ReferenceKind::InstanceSubgraph);
    if (r.all_of<ArmatureModifier>(e)) add(ReferenceKind::Skin);
    if (const auto *constraints = r.try_get<const BoneConstraints>(e)) {
        for (uint32_t i = 0u; i < constraints->Stack.size(); ++i) add(ReferenceKind::Constraint, i);
    }
    if (const auto *collider = r.try_get<const ColliderShape>(e); collider && IsMeshBackedShape(collider->Shape)) add(ReferenceKind::ColliderMesh);
    if (r.all_of<ColliderMaterial>(e)) {
        add(ReferenceKind::PhysicsMaterial);
        add(ReferenceKind::CollisionFilter);
    }
    if (r.all_of<PhysicsJoint>(e)) {
        add(ReferenceKind::JointBody);
        add(ReferenceKind::JointDescription);
    }
    if (const auto *trigger = r.try_get<const TriggerNodes>(e)) {
        add(ReferenceKind::TriggerFilter);
        add_list(ReferenceKind::TriggerMember, trigger->Nodes.size());
    }
    if (const auto *filter = r.try_get<const ::CollisionFilter>(e)) {
        add_list(ReferenceKind::FilterSystem, filter->Systems.size());
        if (filter->Mode != CollideMode::All) add_list(ReferenceKind::FilterCollideSystem, filter->CollideSystems.size());
    }
    // An object shows the material slots of the geometry it draws, after its own references.
    for (uint32_t i = 0; i < MaterialSlots(r, geometry).size(); ++i) add(ReferenceKind::Material, i);
}

// The component required on the target of an entity reference.
decltype(auto) ReferenceTarget(ReferenceKind kind, auto &&visit) {
    using enum ReferenceKind;
    switch (kind) {
        case InstanceSubgraph: return visit.template operator()<SubgraphItems>();
        case Parent:
        case Geometry:
        case ColliderMesh: return visit.template operator()<ObjectKind>();
        case Skin: return visit.template operator()<ArmatureObject>();
        case PhysicsMaterial: return visit.template operator()<::PhysicsMaterial>();
        case CollisionFilter:
        case TriggerFilter: return visit.template operator()<::CollisionFilter>();
        case JointDescription: return visit.template operator()<::JointDescription>();
        case FilterSystem:
        case FilterCollideSystem: return visit.template operator()<CollisionSystem>();
        case Constraint:
        case JointBody:
        case TriggerMember: return visit.template operator()<Transform>();
        case Material: return visit.template operator()<::Material>();
        case Texture: return visit.template operator()<MaterializedTexture>();
    }
    std::unreachable();
}

bool Accepts(const state::Registry &r, SceneReference reference, state::Entity e) {
    using enum ReferenceKind;
    if (e == state::Null) return true;
    if (!r.valid(e) || !ReferenceTarget(reference.Kind, [&]<typename T> { return r.all_of<T>(e); })) return false;
    if (reference.Kind == InstanceSubgraph) {
        // The subgraph holds neither the object of the Instance nor an object below it.
        if (!IsSceneSubgraph(r, e) || UseRecurses(r, e, SubgraphHolding(r, reference.Owner))) return false;
        return std::ranges::none_of(ObjectClosure(r, reference.Owner), [&](state::Entity below) { return !r.all_of<CopyOf>(below) && SubgraphCopies(r, e, below); });
    }
    if (reference.Kind == Parent) return !r.any_of<BoneIndex, SubElementOf>(e);
    if (reference.Kind == Skin) return r.get<const ArmatureObject>(e).Entity == r.get<const ArmatureModifier>(reference.Owner).ArmatureEntity;
    if (reference.Kind == Geometry) return e != reference.Owner && DrawnGeometry(r, e) != state::Null;
    if (reference.Kind == ColliderMesh) return DrawnGeometry(r, e) != state::Null;
    return true;
}

bool HasReference(const state::Registry &r, SceneReference reference) {
    bool found = false;
    ForEachReference(r, reference.Owner, [&](auto candidate) { found |= candidate == reference; });
    return found;
}
// The value of a reference its owner has, as the owner's components and buffers hold it.
// A geometry or collider geometry reference holds the geometry.
state::Entity StoredValue(const state::Registry &r, SceneReference reference) {
    const auto e = reference.Owner;
    const auto index = reference.Index;
    if (const auto *list = ReferenceList(r, reference)) return index < list->size() ? (*list)[index] : state::Null;
    using enum ReferenceKind;
    switch (reference.Kind) {
        case InstanceSubgraph: return r.get<const Instance>(e).SubgraphEntity;
        case Parent: return ParentOrNull(r, e);
        case Geometry: return DrawnGeometry(r, e);
        case Skin: return r.get<const ArmatureModifier>(e).ArmatureObjectEntity;
        case Constraint: return r.get<const BoneConstraints>(e).Stack[index].TargetEntity;
        case ColliderMesh: return r.get<const ColliderShape>(e).MeshEntity;
        case PhysicsMaterial: return r.get<const ::ColliderMaterial>(e).PhysicsMaterialEntity;
        case CollisionFilter: return r.get<const ::ColliderMaterial>(e).CollisionFilterEntity;
        case JointBody: return r.get<const PhysicsJoint>(e).ConnectedNode;
        case JointDescription: return r.get<const PhysicsJoint>(e).JointDescriptionEntity;
        case TriggerFilter: return r.get<const TriggerNodes>(e).CollisionFilterEntity;
        case Material: return MaterialAt(r, MaterialSlots(r, DrawnGeometry(r, e))[index]);
        case Texture: {
            uint32_t slot;
            std::memcpy(&slot, reinterpret_cast<const std::byte *>(&GetMaterials(r)[r.get<const ::Material>(e).Index]) + index, sizeof(slot));
            return TextureAt(r, slot);
        }
        case TriggerMember:
        case FilterSystem:
        case FilterCollideSystem: std::unreachable();
    }
    std::unreachable();
}
// The item a stored value shows as: the object that stands for a geometry, or the value itself.
state::Entity Resolved(const state::Registry &r, SceneReference reference, state::Entity stored, const GeometryDrawers &drawers) {
    const auto e = reference.Owner, geometry = stored;
    if (reference.Kind == ReferenceKind::Geometry) {
        const auto first = FirstDrawer(drawers, geometry);
        return first == e ? state::Null : first;
    }
    if (reference.Kind == ReferenceKind::ColliderMesh) {
        if (geometry == state::Null || geometry == DrawnGeometry(r, e)) return state::Null;
        const auto first = FirstDrawer(drawers, geometry);
        return first != state::Null ? first : geometry;
    }
    return stored;
}
// Appends the owner's references with their stored values, except an object's material slots, which its geometry's node holds.
void StoredReferences(const state::Registry &r, state::Entity owner, std::vector<ItemReference> &stored) {
    ForEachReference(r, owner, [&](auto reference) {
        if (reference.Kind != ReferenceKind::Material) stored.push_back({reference.Kind, reference.Index, StoredValue(r, reference)});
    });
}
// An entity with one of these components has a node: the kinds of item a page shows, and a geometry for its material slots.
using ItemKinds = TypeList<ObjectKind, BoneIndex, Scene, ::PhysicsMaterial, CollisionSystem, ::CollisionFilter, JointDescription, SubgraphItems, GeometryGraph>;
bool HasNode(const state::Registry &r, state::Entity e) {
    return []<typename... Ks>(const state::Registry &r, state::Entity e, TypeList<Ks...>) { return r.any_of<Ks...>(e); }(r, e, ItemKinds{});
}
bool CanExpose(const state::Registry &r, std::span<const state::Entity> items, const SubgraphPort &port) {
    if (std::ranges::contains(items, std::get<PromotedField>(port.Binding).Entity)) return true;
    for (const auto item : items) {
        if (const auto *ports = r.try_get<const SubgraphInterface>(item)) {
            if (std::ranges::any_of(ports->Ports, [&](const auto &inner) { return port.Binding == inner.Binding; })) return true;
        }
    }
    return false;
}
// "BaseColorTexture.Slot" reads "Base color texture".
std::string TextureLabel(uint32_t offset) {
    std::string_view path{action::FieldAt<PBRMaterial>(uint16_t(offset)).Path};
    if (path.ends_with(".Slot")) path.remove_suffix(5);
    std::string label;
    for (const char c : path) {
        const bool upper = std::isupper(static_cast<unsigned char>(c));
        if (c == '.' || (upper && !label.empty() && label.back() != ' ')) label += ' ';
        if (c != '.') label += upper && !label.empty() ? char(std::tolower(static_cast<unsigned char>(c))) : c;
    }
    return label;
}
} // namespace

GeometryDrawers DrawersOf(const state::Registry &r) {
    GeometryDrawers drawers;
    for (const auto [object, geometry] : r.view<const ObjectGeometry>().each()) {
        if (r.all_of<ObjectKind>(object) && r.all_of<GeometryGraph>(geometry.Entity)) drawers.emplace_back(geometry.Entity, object);
    }
    std::ranges::sort(drawers);
    const auto [first, last] = std::ranges::unique(drawers, {}, &GeometryDrawers::value_type::first);
    drawers.erase(first, last);
    return drawers;
}

state::Entity ReferenceValue(const state::Registry &r, SceneReference reference) {
    if (!HasReference(r, reference)) return state::Null;
    const bool draws = reference.Kind == ReferenceKind::Geometry || reference.Kind == ReferenceKind::ColliderMesh;
    return Resolved(r, reference, StoredValue(r, reference), draws ? DrawersOf(r) : GeometryDrawers{});
}

std::string ItemName(const state::Registry &r, state::Entity entity) {
    if (entity == state::Document) return "Document";
    if (const auto *material = r.try_get<const ::Material>(entity)) return material->Name.empty() ? "Material" : material->Name;
    if (const auto *texture = r.try_get<const MaterializedTexture>(entity)) return texture->Params.Name.empty() ? "Texture" : texture->Params.Name;
    // A geometry is named only as a collider's value, when no object draws it.
    if (r.all_of<GeometryGraph>(entity)) return "Separate mesh";
    if (const auto *m = r.try_get<const ::PhysicsMaterial>(entity)) return m->Name;
    if (const auto *f = r.try_get<const ::CollisionFilter>(entity)) return f->Name;
    if (const auto *c = r.try_get<const CollisionSystem>(entity)) return c->Name;
    if (const auto *j = r.try_get<const JointDescription>(entity)) return j->Name;
    return GetName(r, entity);
}

namespace {
ItemNode DeriveNode(const state::Registry &r, state::Entity e) {
    ItemNode node{.Title = ItemName(r, e)};
    StoredReferences(r, e, node.References);
    const auto slots = MaterialSlots(r, r.all_of<GeometryGraph>(e) ? e : state::Null);
    const auto materials = slots.empty() ? std::vector<state::Entity>{} : MaterialsByIndex(r);
    for (uint32_t i = 0; i < slots.size(); ++i) node.References.push_back({ReferenceKind::Material, i, slots[i] < materials.size() ? materials[slots[i]] : state::Null});
    return node;
}
} // namespace

void ShownReferences(const state::Registry &r, state::Entity item, const GeometryDrawers &drawers, std::vector<ItemReference> &shown) {
    shown.clear();
    if (!r.valid(item)) return;
    // A material's texture slots are read from its record, which no stored node holds.
    if (r.all_of<::Material>(item)) {
        StoredReferences(r, item, shown);
    } else if (const auto *node = r.try_get<const ItemNode>(item)) {
        shown = node->References;
        for (const auto &reference : node->References) {
            const auto *geometry = reference.Kind == ReferenceKind::Geometry ? r.try_get<const ItemNode>(reference.Value) : nullptr;
            if (geometry) shown.append_range(geometry->References);
        }
    }
    for (auto &reference : shown) reference.Value = Resolved(r, {item, reference.Kind, reference.Index}, reference.Value, drawers);
}

bool NodesSettled(const state::Registry &r) {
    bool settled = true;
    for (const auto e : r.view<const ItemNode>()) settled &= HasNode(r, e) && r.get<const ItemNode>(e) == DeriveNode(r, e);
    const auto kind = [&]<typename K> { for (const auto e : r.view<const K>()) settled &= r.all_of<ItemNode>(e); };
    [&]<typename... Ks>(TypeList<Ks...>) { (kind.template operator()<Ks>(), ...); }(ItemKinds{});
    return settled;
}

std::string ReferenceLabel(const state::Registry &r, SceneReference reference) {
    using enum ReferenceKind;
    if (reference.Kind == Texture) return TextureLabel(reference.Index);
    constexpr std::array labels{"Parent", "Geometry", "Skin armature", "Constraint target", "Collider geometry", "Physics material", "Collision filter", "Joint body", "Joint description", "Trigger filter", "Trigger member", "Material slot", "Texture", "Subgraph", "Member system", "Collide system"};
    std::string label{labels[uint8_t(reference.Kind)]};
    if (reference.Kind == FilterCollideSystem && r.get<const ::CollisionFilter>(reference.Owner).Mode == CollideMode::Blocklist) label = "Blocked system";
    if (const auto *list = ReferenceList(r, reference)) {
        if (reference.Index < list->size()) return label + " " + std::to_string(reference.Index);
        label.front() = char(std::tolower(static_cast<unsigned char>(label.front())));
        return "Add " + label;
    }
    if (reference.Kind == Constraint || reference.Kind == Material) label += " " + std::to_string(reference.Index);
    return label;
}

bool ReferenceAccepts(const state::Registry &r, SceneReference reference, state::Entity source) {
    return HasReference(r, reference) && Accepts(r, reference, source);
}

std::vector<state::Entity> ReferenceCandidates(const state::Registry &r, SceneReference reference) {
    std::vector<state::Entity> candidates;
    if (!HasReference(r, reference)) return candidates;
    ReferenceTarget(reference.Kind, [&]<typename T> {
        for (const auto e : r.view<const T>())
            if (Accepts(r, reference, e)) candidates.push_back(e);
    });
    std::ranges::sort(candidates);
    return candidates;
}

std::vector<SceneReference> ReferencesTo(const state::Registry &r, state::Entity item) {
    std::vector<SceneReference> users;
    std::vector<ItemReference> shown;
    const auto drawers = DrawersOf(r);
    const auto visit = [&](state::Entity owner) {
        ShownReferences(r, owner, drawers, shown);
        for (const auto &reference : shown) {
            if (reference.Value == item) users.push_back({owner, reference.Kind, reference.Index});
        }
    };
    for (const auto e : r.view<const ::Material>()) visit(e);
    // A geometry's node holds the material slots that each object drawing it shows.
    for (const auto e : r.view<const ItemNode>()) {
        if (!r.all_of<GeometryGraph>(e)) visit(e);
    }
    std::ranges::sort(users);
    return users;
}
} // namespace graph

namespace action::nodes {
using namespace graph;
void Apply(state::Registry &r, const AssignReference &a) {
    if (!ReferenceAccepts(r, a.Reference, a.Source)) return Fail(r, "The source does not match this scene input");
    // An instance's subgraph is its own, and any other reference of a copy is its source's.
    const auto [e, source] = a.Reference.Kind == ReferenceKind::InstanceSubgraph ? std::pair{a.Reference.Owner, a.Source} : CopyReferenceTarget(r, a.Reference.Owner, a.Source);
    const auto index = a.Reference.Index;
    // A geometry the object stops using is removed once no object or collider uses it.
    const auto release = [&](state::Entity geometry) { r.Resources.emplace<PendingObjectRemovals>().Buffers.emplace(geometry); };
    using enum ReferenceKind;
    switch (a.Reference.Kind) {
        case InstanceSubgraph:
            if (source == state::Null) MakeInstanceUnique(r, e);
            else SetInstanceSubgraph(r, e, source);
            return;
        case Parent:
            if (source == state::Null) object::Apply(r, object::ClearParent{e});
            else object::Apply(r, object::SetParent{e, source});
            break;
        case Geometry: {
            // Clearing gives the object its own copy of a geometry another object also draws.
            const auto drawn = r.get<const ObjectGeometry>(e).Entity;
            if (source == state::Null) {
                bool shared = false;
                for (const auto [object, geometry] : r.view<const ObjectGeometry>().each()) shared |= object != e && geometry.Entity == drawn;
                if (shared) r.replace<ObjectGeometry>(e, object::CopyGeometry(r, drawn));
                break;
            }
            const auto geometry = DrawnGeometry(r, source);
            if (!ObjectReferencesAcyclic(r, {}, state::Null, state::Null, std::pair{e, geometry})) return Fail(r, "The geometry assignment would create a dependency cycle");
            r.replace<ObjectGeometry>(e, geometry);
            release(drawn);
            break;
        }
        case Skin: r.patch<ArmatureModifier>(e, [&](auto &modifier) { modifier.ArmatureObjectEntity = source; }); break;
        case Constraint: bone::Apply(r, bone::SetConstraintTarget{index, source, e}); break;
        case ColliderMesh:
            release(r.get<const ColliderShape>(e).MeshEntity);
            r.patch<ColliderShape>(e, [&](auto &shape) { shape.MeshEntity = DrawnGeometry(r, source == state::Null ? e : source); });
            break;
        case PhysicsMaterial: action::Apply(r, Core{UpdateOf<&ColliderMaterial::PhysicsMaterialEntity>(e, source)}); break;
        case CollisionFilter: action::Apply(r, Core{UpdateOf<&ColliderMaterial::CollisionFilterEntity>(e, source)}); break;
        case JointBody: action::Apply(r, Core{UpdateOf<&PhysicsJoint::ConnectedNode>(e, source)}); break;
        case JointDescription: action::Apply(r, Core{UpdateOf<&PhysicsJoint::JointDescriptionEntity>(e, source)}); break;
        case TriggerFilter: action::Apply(r, Core{UpdateOf<&TriggerNodes::CollisionFilterEntity>(e, source)}); break;
        // A list reference replaces its entry, or removes it for no source, and the reference past the last entry adds one.
        case TriggerMember:
            r.patch<TriggerNodes>(e, [&](auto &trigger) {
                if (index < trigger.Nodes.size()) trigger.Nodes.erase(trigger.Nodes.begin() + index);
                if (source != state::Null && !std::ranges::contains(trigger.Nodes, source)) trigger.Nodes.push_back(source);
            });
            break;
        case FilterSystem:
        case FilterCollideSystem: {
            using Toggle = physics::ToggleFilterEntity;
            const auto which = a.Reference.Kind == FilterSystem ? Toggle::List::Systems : Toggle::List::CollideSystems;
            const auto &list = *ReferenceList(r, a.Reference);
            if (index < list.size()) physics::Apply(r, Toggle{e, which, state::Entity{list[index]}, false});
            if (source != state::Null) physics::Apply(r, Toggle{e, which, source, true});
            break;
        }
        case Material: object::Apply(r, object::SetMaterialAssignment{index, source, DrawnGeometry(r, e)}); break;
        case Texture: {
            const auto slot = source == state::Null ? InvalidSlot : r.get<const MaterializedTexture>(source).SamplerSlot;
            object::Apply(r, object::UpdateMaterial<uint32_t>{e, uint16_t(index), slot});
            break;
        }
    }
}

void Apply(state::Registry &r, const ExposePort &a) {
    SubgraphPort port{.Name = a.Name ? *a.Name : std::string{}, .Binding = a.Field};
    const bool valid = !port.Name.empty() && action::ReadUpdate(r, a.Field.Entity, a.Field.Component, a.Field.Offset);
    if (!valid || !r.all_of<SubgraphItems>(a.Subgraph) || !CanExpose(r, GraphItems(r, a.Subgraph), port)) return Fail(r, "Expose a field of an item inside the subgraph, through each enclosing subgraph");
    const auto &interface = r.get<const SubgraphInterface>(a.Subgraph);
    if (std::ranges::any_of(interface.Ports, [&](const auto &existing) { return port.Binding == existing.Binding; })) return;
    if (interface.NextId >= UINT16_MAX) return Fail(r, "The subgraph has no available port identities");
    r.patch<SubgraphInterface>(a.Subgraph, [&](auto &interface) {
        port.Id = uint16_t(interface.NextId++);
        interface.Ports.push_back(std::move(port));
    });
}
} // namespace action::nodes

namespace graph {
void DropUnreachablePorts(state::Registry &r, state::Entity root) {
    const auto subgraphs = SortedEntities(r.view<const SubgraphItems>());
    const auto order = DependencyOrder(uint32_t(subgraphs.size()), [&](auto edge) {
        for (uint32_t child = 0; child < subgraphs.size(); ++child) {
            const auto at = std::ranges::lower_bound(subgraphs, r.get<const SubgraphItems>(subgraphs[child]).ParentEntity);
            if (at != subgraphs.end() && *at == r.get<const SubgraphItems>(subgraphs[child]).ParentEntity) edge(child, uint32_t(at - subgraphs.begin()));
        }
    });
    for (const auto index : order) {
        const auto subgraph = subgraphs[index];
        if (GraphRoot(r, subgraph) != root) continue;
        const auto items = GraphItems(r, subgraph);
        const auto reachable = [&](const SubgraphPort &port) { return CanExpose(r, items, port); };
        if (!std::ranges::all_of(r.get<const SubgraphInterface>(subgraph).Ports, reachable)) r.patch<SubgraphInterface>(subgraph, [&](auto &interface) { std::erase_if(interface.Ports, std::not_fn(reachable)); });
    }
}

void SettleItemGraphs(state::Registry &r, bool restoring) {
    auto &changes = r.Resources.get<GraphChanges>();
    bool changed = !changes.Pages.empty();
    std::vector<state::Entity> dead;
    for (const auto e : changes.Nodes) {
        if (!r.valid(e)) {
            dead.push_back(e);
        } else if (!HasNode(r, e)) {
            // A geometry page titles a subgraph node by its subgraph's name, and titles and labels a node by its operation.
            changed |= r.remove<ItemNode>(e) > 0u || r.all_of<GraphNode>(e) || r.all_of<Subgraph, GraphNodes>(e);
        } else if (auto node = DeriveNode(r, e); !r.all_of<ItemNode>(e) || r.get<const ItemNode>(e) != node) {
            r.emplace_or_replace<ItemNode>(e, std::move(node));
            changed = true;
        }
    }
    for (const auto e : changes.Pages) {
        if (!r.valid(e)) dead.push_back(e);
    }
    if (changed || !dead.empty()) ++changes.Revision;
    if (!changes.Cells.empty()) ++changes.CellsRevision;
    const bool fields = !changes.Fields.empty();
    for (auto *set : {&changes.Nodes, &changes.Pages, &changes.Cells, &changes.Fields}) set->clear();
    // A restore brings back cells, items and ports that already match its entities.
    // A pass that destroyed no entity and no component a port can address leaves them as they are.
    if (restoring || (dead.empty() && !fields)) return;
    SortUnique(dead);
    const auto is_dead = [&](state::Entity item) { return std::ranges::binary_search(dead, item); };
    for (const auto graph : SortedEntities(r.view<const GraphLayout>())) {
        const auto &cells = r.get<const GraphLayout>(graph).Cells;
        if (std::ranges::none_of(cells, is_dead, &GraphCell::Item)) continue;
        if (std::ranges::all_of(cells, is_dead, &GraphCell::Item)) r.remove<GraphLayout>(graph);
        else r.patch<GraphLayout>(graph, [&](auto &layout) { std::erase_if(layout.Cells, [&](const auto &cell) { return is_dead(cell.Item); }); });
    }
    for (const auto subgraph : SortedEntities(r.view<const SubgraphItems>())) {
        if (std::ranges::any_of(r.get<const SubgraphItems>(subgraph).Items, is_dead)) r.patch<SubgraphItems>(subgraph, [&](auto &edited) { std::erase_if(edited.Items, is_dead); });
        // A value port lasts while its field exists.
        const auto exists = [&](const SubgraphPort &port) {
            const auto &field = std::get<PromotedField>(port.Binding);
            return bool(action::ReadUpdate(r, field.Entity, field.Component, field.Offset));
        };
        if (!std::ranges::all_of(r.get<const SubgraphInterface>(subgraph).Ports, exists)) r.patch<SubgraphInterface>(subgraph, [&](auto &interface) { std::erase_if(interface.Ports, std::not_fn(exists)); });
    }
}
} // namespace graph
