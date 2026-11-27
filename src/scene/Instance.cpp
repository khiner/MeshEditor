#include "scene/Instance.h"

#include "Path.h"
#include "SortUnique.h"
#include "action/Core.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "animation/Clips.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioSamples.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "audio/RealImpactComponents.h"
#include "graph/Graph.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/ObjectGeometry.h"
#include "render/RenderChanges.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "state/EntityReferences.h"

#include <format>
#include <stdexcept>

namespace {
using Values = TypeList<Transform, Name, Path, ObjectKind, ObjectGeometry, Visibility, ScaleLocked, Perspective, Orthographic, SceneLight, ArmatureObject, BoneJoint, BoneSubPartOf, BoneJointEntities, SubElementOf, ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, BoneDelta, BoneConstraints, PhysicsMotion, PhysicsVelocity, ColliderShape, ColliderMaterial, ColliderPolicy, TriggerTag, TriggerNodes, PhysicsJoint, Striker, ModalSoundControls, ContactSurface, SurfaceSoundControls, AcousticMaterial, SoundVerticesModel, ModalGain, ModalTuning, ModalSolveSettings, MassProperties, RealImpactMicrophone, RealImpactActiveMicrophone>;
// What a copy reads through its source, and holds itself once it stops being a copy.
using Shared = TypeList<AnimationClips, VertexSamples, ModalModes, ModalEigenSummary, RealImpactVertices>;

using Change = InstanceChanges::Change;

// A restore brings back instances as they were published.
template<typename T> void Changed(state::Registry &r, state::Entity e) {
    auto &changes = r.Resources.get<InstanceChanges>();
    if (r.Restoring) return;
    if constexpr (std::same_as<T, Instance> || std::same_as<T, graph::InstanceOnPoints>) changes.Uses = true;
    if (!changes.Publishing && !r.view<const Instance>().empty()) changes.Values.emplace_back(e, state::Slot<T>);
}
// An action writes a value a copy shows to the copy's source, or to its instance's port, and publication writes the copy.
template<typename T> void Written(state::Registry &r, state::Entity e) {
    if (r.Applying && !r.Restoring && !r.Resources.get<const InstanceChanges>().Publishing && r.all_of<CopyOf>(e)) throw std::logic_error(std::format("Action write to an instance's copy: {}", state::LeafName<T>()));
    Changed<T>(r, e);
}

template<typename T> void Copy(state::Registry &r, const SourceCopy &pair, std::span<const SourceCopy> copies) {
    if (const auto *value = r.try_get<const T>(pair.SourceEntity)) {
        T copied = *value;
        state::ForEachEntity(copied, [&](auto &e) { e = CopyFor(copies, e); });
        if constexpr (std::same_as<T, ArmatureObject>) {
            if (const auto *prior = r.try_get<const ArmatureObject>(pair.CopyEntity); prior && (prior->Entity != copied.Entity || prior->BoneEntities.size() != copied.BoneEntities.size())) r.remove<ArmaturePoseState>(pair.CopyEntity);
        }
        r.emplace_or_replace<T>(pair.CopyEntity, std::move(copied));
    } else r.remove<T>(pair.CopyEntity);
}

void CopyHelperGeometry(state::Registry &r, const SourceCopy &pair) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (!r.all_of<VertexStoreId>(pair.SourceEntity)) {
        r.remove<VertexStoreId, ModelsBuffer>(pair.CopyEntity);
        return;
    }
    if (r.all_of<VertexStoreId>(pair.CopyEntity)) return;
    if (r.all_of<ArmatureObject>(pair.SourceEntity)) {
        const auto data = primitive::BoneOctahedron(1.f);
        r.emplace<VertexStoreId>(pair.CopyEntity, meshes.AllocateVertexBuffer(data.Mesh.Positions, data.Attrs));
    } else if (r.all_of<BoneJoint>(pair.SourceEntity)) {
        const auto data = primitive::BoneSphereDisc();
        r.emplace<VertexStoreId>(pair.CopyEntity, meshes.AllocateVertexBuffer(data.Mesh.Positions, {}));
    }
}

// Writes each port value the instance holds to the field of the copy whose source the port promotes.
void SetPortValues(state::Registry &r, const Instance &instance, std::span<const SourceCopy> copies) {
    if (instance.PortValues.empty()) return;
    const auto &ports = r.get<const graph::SubgraphInterface>(instance.SubgraphEntity).Ports;
    for (const auto &own : instance.PortValues) {
        const auto port = std::ranges::find(ports, own.PortId, &graph::SubgraphPort::Id);
        const auto *field = port != ports.end() ? std::get_if<graph::PromotedField>(&port->Binding) : nullptr;
        if (!field) continue;
        if (const auto copy = CopyFor(copies, field->Entity); copy != field->Entity) action::WriteField(r, copy, field->Component, field->Offset, own.Value.Words);
    }
}

// Brings the holder's copies of the closure's entities up to date with the changes, and returns the copies that joined or left.
// A holder with an Instance stores the copies there and has its port values set on them.
// A copy whose source's parent has a copy hangs below that copy, and any other object's copy hangs below `top`, or has no parent for none.
// `resync` copies every value again.
std::vector<state::Entity> Publish(state::Registry &r, state::Entity holder, state::Entity top, std::vector<SourceCopy> &copies, std::span<const state::Entity> closure, std::span<const Change> changes, bool resync = false) {
    const auto before = copies;
    const auto removed = ReconcileCopies(r, copies, closure);
    const bool reshaped = copies != before;
    const bool structure = reshaped || std::ranges::any_of(changes, [&](const auto &c) {
                               return c.Type == state::Slot<SceneParent> && std::ranges::binary_search(closure, c.Entity);
                           });
    if (reshaped && r.all_of<Instance>(holder)) r.patch<Instance>(holder, [&](auto &held) { held.Copies = copies; });
    auto moved = removed;
    for (const auto &pair : copies) {
        const bool fresh = !r.all_of<CopyOf>(pair.CopyEntity);
        if (fresh) {
            r.emplace<CopyOf>(pair.CopyEntity, holder, pair.SourceEntity);
            moved.push_back(pair.CopyEntity);
        }
        [&]<typename... Ts>(TypeList<Ts...>) {
            ([&] {
                if (fresh || resync || structure || std::ranges::binary_search(changes, Change{pair.SourceEntity, state::Slot<Ts>})) Copy<Ts>(r, pair, copies);
            }(),
             ...);
        }(Values{});
        CopyHelperGeometry(r, pair);
        if (fresh || std::ranges::binary_search(changes, Change{pair.SourceEntity, state::Slot<MorphWeightRange>})) {
            const auto *weights = r.try_get<const MorphWeightRange>(pair.SourceEntity);
            const auto *current = r.try_get<const MorphWeightRange>(pair.CopyEntity);
            auto &buffer = r.Resources.get<GpuBuffers>().MorphWeightBuffer;
            if (!weights) r.remove<MorphWeightRange>(pair.CopyEntity);
            else if (!current || current->Weights.Count != weights->Weights.Count) {
                r.remove<MorphWeightRange>(pair.CopyEntity);
                r.emplace<MorphWeightRange>(pair.CopyEntity, buffer.Clone(weights->Weights));
            } else {
                const auto source_values = buffer.Get(weights->Weights);
                if (!std::ranges::equal(source_values, buffer.Get(current->Weights))) std::ranges::copy(source_values, buffer.GetMutable(current->Weights).begin());
            }
            r.Resources.get<RenderChanges>().MorphWeights.emplace(pair.CopyEntity);
        }
    }
    if (structure) {
        std::vector<state::Entity> children;
        for (const auto &pair : copies) children.push_back(pair.CopyEntity);
        ClearParents(r, children);
        for (const auto &pair : copies) {
            const auto parent = ParentOrNull(r, pair.SourceEntity);
            if (const auto mapped = CopyFor(copies, parent); mapped != parent) SetParent(r, pair.CopyEntity, mapped);
            else if (top != state::Null && r.all_of<ObjectKind>(pair.SourceEntity)) SetParent(r, pair.CopyEntity, top);
            r.remove<SceneMembership>(pair.CopyEntity);
        }
    }
    if (const auto *instance = r.try_get<const Instance>(holder)) SetPortValues(r, *instance, copies);
    // A removed copy that the subgraph of another instance holds is no longer a copy of this one.
    for (const auto e : removed) r.remove<CopyOf>(e);
    Destroy(r, removed);
    return moved;
}
// Publishes without recording the changes the publication makes.
std::vector<state::Entity> PublishNow(state::Registry &r, state::Entity holder, state::Entity top, std::vector<SourceCopy> &copies, std::span<const state::Entity> closure) {
    auto &changes = r.Resources.get<InstanceChanges>();
    const bool publishing = std::exchange(changes.Publishing, true);
    auto moved = Publish(r, holder, top, copies, closure, {});
    changes.Publishing = publishing;
    return moved;
}
// Gives the object an Instance of the scene subgraph with the port values, and makes its copies.
void Attach(state::Registry &r, state::Entity object, state::Entity subgraph, std::vector<graph::PortValue> values = {}) {
    r.emplace<Instance>(object, subgraph, std::move(values));
    std::vector<SourceCopy> copies;
    PublishNow(r, object, object, copies, SubgraphClosure(r, subgraph));
}
// Whether the subgraph is the other or a subgraph inside it.
bool Within(const state::Registry &r, state::Entity subgraph, state::Entity other) {
    for (; r.valid(subgraph) && r.all_of<graph::SubgraphItems>(subgraph); subgraph = r.get<const graph::SubgraphItems>(subgraph).ParentEntity)
        if (subgraph == other) return true;
    return false;
}
} // namespace

void Reset(InstanceChanges &changes, state::Registry &) {
    changes.Values.clear();
    changes.Publishing = changes.Uses = false;
}

bool IsSceneSubgraph(const state::Registry &r, state::Entity e) {
    if (!r.valid(e) || !r.all_of<graph::SubgraphItems>(e)) return false;
    const auto root = graph::GraphRoot(r, e);
    return r.valid(root) && (r.all_of<Scene>(root) || (root == state::Document && !SubgraphClosure(r, e).empty()));
}

state::Entity AddInstance(state::Registry &r, state::Entity subgraph, const ObjectCreateInfo &info, std::vector<graph::PortValue> values) {
    const auto object = AddEmpty(r, info);
    Attach(r, object, subgraph, std::move(values));
    return object;
}

state::Entity AddGeneratedInstance(state::Registry &r, state::Entity parent, state::Entity subgraph, const Transform &placement) {
    const auto object = r.create();
    r.emplace<Transform>(object, placement);
    SetParent(r, object, parent);
    EmplaceObject(r, object, ObjectType::Empty);
    r.emplace<Visibility>(object);
    EmplaceUniqueName(r, object, NameStem(GetName(r, subgraph)));
    Attach(r, object, subgraph);
    return object;
}

void SetInstanceSubgraph(state::Registry &r, state::Entity object, state::Entity subgraph) {
    const auto &current = r.get<const Instance>(object);
    if (current.SubgraphEntity == subgraph) return;
    std::vector<state::Entity> removed;
    for (const auto &pair : current.Copies) removed.push_back(pair.CopyEntity);
    r.remove<Instance>(object);
    for (const auto e : removed) r.remove<CopyOf>(e);
    Destroy(r, removed);
    Attach(r, object, subgraph);
}

void PublishSubgraphInstances(state::Registry &r) {
    auto &changes = r.Resources.get<InstanceChanges>();
    for (const auto e : r.Resources.get<const RenderChanges>().MorphWeights) {
        if (!r.all_of<CopyOf>(e)) changes.Values.emplace_back(e, state::Slot<MorphWeightRange>);
    }
    if (changes.Values.empty() || r.view<const Instance>().empty()) {
        changes.Values.clear();
        return;
    }
    auto values = std::move(changes.Values);
    SortUnique(values);
    const auto instances = SortedEntities(r.view<const Instance>());
    // An instance publishes after each instance whose copies it copies.
    std::vector<state::Entity> ordered, visiting;
    const auto visit = [&](auto &&self, state::Entity instance) -> void {
        if (std::ranges::contains(ordered, instance) || std::ranges::contains(visiting, instance)) return;
        visiting.push_back(instance);
        for (const auto &pair : r.get<const Instance>(instance).Copies) {
            if (r.all_of<Instance>(pair.SourceEntity)) self(self, pair.SourceEntity);
            else if (const auto *binding = r.try_get<const CopyOf>(pair.SourceEntity); binding && binding->ObjectEntity != instance) self(self, binding->ObjectEntity);
        }
        visiting.pop_back();
        ordered.push_back(instance);
    };
    for (const auto instance : instances) visit(visit, instance);
    // The subgraphs whose items changed, and the instances that copied every value again.
    std::vector<state::Entity> regrouped, resynced;
    for (const auto &change : values)
        if (change.Type == state::Slot<graph::SubgraphItems>) regrouped.push_back(change.Entity);
    changes.Publishing = true;
    for (const auto instance : ordered) {
        if (!r.all_of<Instance>(instance)) continue;
        auto copies = r.get<const Instance>(instance).Copies;
        const auto subgraph = r.get<const Instance>(instance).SubgraphEntity;
        const auto affected = [&](const auto &change) {
            return CopyFor(copies, change.Entity) != change.Entity || (change.Type == state::Slot<SceneParent> && CopyFor(copies, ParentOrNull(r, change.Entity)) != ParentOrNull(r, change.Entity));
        };
        // A changed interface, or a source instance that copied every value again, copies every value again.
        const bool resync = std::ranges::binary_search(values, Change{subgraph, state::Slot<graph::SubgraphInterface>}) || std::ranges::any_of(copies, [&](const auto &pair) {
                                const auto *source = r.try_get<const CopyOf>(pair.SourceEntity);
                                return source && std::ranges::contains(resynced, source->ObjectEntity);
                            });
        const bool revalued = std::ranges::binary_search(values, Change{instance, state::Slot<Instance>});
        const bool reshaped = std::ranges::any_of(regrouped, [&](state::Entity changed) { return Within(r, changed, subgraph); });
        if (!resync && !revalued && !reshaped && !std::ranges::any_of(values, affected)) continue;
        // A use that set or cleared a port value copies each promoted field again before the values it still holds apply.
        std::vector<Change> revalues;
        if (revalued) {
            revalues = values;
            for (const auto &port : r.get<const graph::SubgraphInterface>(subgraph).Ports) {
                const auto *field = std::get_if<graph::PromotedField>(&port.Binding);
                if (const auto type = field ? action::UpdatedType(field->Component) : std::nullopt) revalues.push_back({field->Entity, *type});
            }
            SortUnique(revalues);
        }
        const auto moved = Publish(r, instance, instance, copies, SubgraphClosure(r, subgraph), revalued ? revalues : values, resync);
        if (resync) resynced.push_back(instance);
        // Each change of a source is a change of its copy for the instances that copy this one's copies.
        const auto count = revalued ? revalues.size() : values.size();
        const auto seen = [&](size_t i) { return revalued ? revalues[i] : values[i]; };
        size_t change = 0;
        for (const auto &m : copies) {
            while (change < count && seen(change).Entity < m.SourceEntity) ++change;
            for (; change < count && seen(change).Entity == m.SourceEntity; ++change) values.push_back({m.CopyEntity, seen(change).Type});
        }
        for (const auto e : moved) values.push_back({e, state::Slot<SceneParent>});
        SortUnique(values);
    }
    changes.Publishing = false;
    changes.Values.clear();
}

void UnplaceSubgraph(state::Registry &r, state::Entity subgraph) {
    const auto closure = SubgraphClosure(r, subgraph);
    const auto held = [&](state::Entity e) { return std::ranges::binary_search(closure, e); };
    // An item leaves the parent it has outside the subgraph, and an object of its own leaves its parent inside.
    std::vector<state::Entity> detached;
    for (const auto e : closure) {
        if (const auto parent = ParentOrNull(r, e); parent != state::Null && !held(parent)) detached.push_back(e);
        const auto *object = r.try_get<const ObjectGeometry>(e);
        const auto *generated = object ? r.try_get<const graph::GraphInstances>(object->Entity) : nullptr;
        for (const auto child : Children{&r, e}) {
            if (!held(child) && (!generated || !std::ranges::contains(generated->Entities, child))) detached.push_back(child);
        }
    }
    ClearParents(r, detached);
    for (const auto e : closure) r.remove<SceneMembership, Selected, Active>(e);
    const auto parent = r.get<const graph::SubgraphItems>(subgraph).ParentEntity;
    if (r.all_of<graph::SubgraphItems>(parent)) r.patch<graph::SubgraphItems>(parent, [&](auto &outer) { std::erase(outer.Items, subgraph); });
    graph::ClearGraphPosition(r, parent, subgraph);
    r.patch<graph::SubgraphItems>(subgraph, [](auto &moved) { moved.ParentEntity = state::Document; });
    r.remove<Selected, Active>(subgraph);
}

void CollectUnplacedSubgraphs(state::Registry &r) {
    if (!std::exchange(r.Resources.get<InstanceChanges>().Uses, false)) return;
    for (;;) {
        std::vector<state::Entity> objects, subgraphs;
        for (const auto [subgraph, held] : r.view<const graph::SubgraphItems>().each()) {
            if (held.ParentEntity != state::Document || SubgraphInstanced(r, subgraph)) continue;
            const auto closure = SubgraphClosure(r, subgraph);
            if (closure.empty()) continue;
            objects.append_range(closure);
            subgraphs.push_back(subgraph);
        }
        if (subgraphs.empty()) break;
        // A subgraph goes with each subgraph inside it.
        for (size_t i = 0; i < subgraphs.size(); ++i) {
            for (const auto item : r.get<const graph::SubgraphItems>(subgraphs[i]).Items)
                if (r.valid(item) && r.all_of<graph::SubgraphItems>(item)) subgraphs.push_back(item);
        }
        SortUnique(objects);
        Destroy(r, objects);
        SortUnique(subgraphs);
        r.destroy(subgraphs);
    }
}

void RegisterInstances(state::Registry &r) {
    r.Resources.emplace<InstanceChanges>();
    const auto track = [&]<typename... Ts>(TypeList<Ts...>) {
        (r.on_construct<Ts, &Changed<Ts>>(), ...);
        (r.on_update<Ts, &Changed<Ts>>(), ...);
        (r.on_destroy<Ts, &Changed<Ts>>(), ...);
    };
    const auto track_values = [&]<typename... Ts>(TypeList<Ts...>) {
        (r.on_construct<Ts, &Written<Ts>>(), ...);
        (r.on_update<Ts, &Written<Ts>>(), ...);
        (r.on_destroy<Ts, &Changed<Ts>>(), ...);
    };
    track_values(Values{});
    track_values(Shared{});
    track(TypeList<SceneParent, MorphWeightRange, Instance>{});
    track(TypeList<graph::InstanceOnPoints, graph::SubgraphInterface, graph::SubgraphItems>{});
}

std::vector<state::Entity> PrepareInstanceRemoval(state::Registry &r, std::span<const state::Entity> entities) {
    std::vector<state::Entity> removed;
    for (const auto e : entities) {
        if (const auto *copies = r.try_get<const Instance>(e))
            for (const auto &pair : copies->Copies) removed.push_back(pair.CopyEntity);
        removed.push_back(e);
    }
    SortUnique(removed);
    return removed;
}

namespace {
// The record a graph's mesh draws, or InvalidOffset for another mesh.
uint32_t GraphRecord(const state::Registry &r, state::Entity mesh) {
    const auto drawn = DrawnStoreId(r, mesh);
    return drawn && r.all_of<graph::GeometryGraph>(mesh) ? *drawn : InvalidOffset;
}
// The records of the graphs the copies draw, once per graph.
void AppendGraphRecords(const state::Registry &r, std::span<const SourceCopy> copies, std::vector<uint32_t> &records) {
    std::vector<state::Entity> copied;
    for (const auto &pair : copies) {
        const auto *geometry = r.try_get<const ObjectGeometry>(pair.CopyEntity);
        if (!geometry || std::ranges::contains(copied, geometry->Entity)) continue;
        if (const auto record = GraphRecord(r, geometry->Entity); record != InvalidOffset) {
            copied.push_back(geometry->Entity);
            records.push_back(record);
        }
    }
}
// Whether a node of the geometry graph, or of a subgraph it uses at any depth, refers to an object that `map` changes.
bool MapsReference(const state::Registry &r, state::Entity graph, auto &&map) {
    for (const auto node : graph::NodesOf(r, graph)) {
        bool maps = false;
        graph::VisitKind(r.get<const graph::GraphNode>(node).Kind, [&]<typename K> {
            if constexpr (graph::ObjectInput<K> != nullptr) {
                const auto object = r.get<const K>(node).*graph::ObjectInput<K>;
                maps = map(object) != object;
            } else if constexpr (std::same_as<K, graph::SubgraphNode>) {
                const auto subgraph = graph::UserSubgraph(r.get<const graph::SubgraphNode>(node).SubgraphId);
                maps = subgraph != state::Null && MapsReference(r, subgraph, map);
            }
        });
        if (maps) return true;
    }
    return false;
}
// Makes the graph refer to what `map` gives for each object it refers to.
// A subgraph node whose subgraph refers to a changed object uses its own copy of that subgraph.
void MapReferences(state::Registry &r, state::Entity graph, auto &&map) {
    for (const auto node : graph::NodesOf(r, graph)) {
        graph::VisitKind(r.get<const graph::GraphNode>(node).Kind, [&]<typename K> {
            if constexpr (graph::ObjectInput<K> != nullptr) {
                if (const auto object = r.get<const K>(node).*graph::ObjectInput<K>; map(object) != object) r.patch<K>(node, [&](auto &p) { p.*graph::ObjectInput<K> = map(object); });
            } else if constexpr (std::same_as<K, graph::SubgraphNode>) {
                const auto subgraph = graph::UserSubgraph(r.get<const graph::SubgraphNode>(node).SubgraphId);
                if (subgraph == state::Null || !MapsReference(r, subgraph, map)) return;
                action::nodes::Apply(r, action::nodes::MakeUnique{node});
                MapReferences(r, graph::UserSubgraph(r.get<const graph::SubgraphNode>(node).SubgraphId), map);
            }
        });
    }
}
// Makes the copies objects of their own.
// Each takes its own copy of the resources it shared with its source, taking each graph's record copy from `clone` in the order of AppendGraphRecords.
// A linked duplicate keeps its source's geometry and armature.
void Detach(state::Registry &r, std::span<const SourceCopy> copies, std::vector<uint32_t>::const_iterator &clone, bool linked) {
    const auto copy_resource = [&](std::vector<SourceCopy> &resources, state::Entity source, auto create) {
        const auto at = std::ranges::lower_bound(resources, source, {}, &SourceCopy::SourceEntity);
        if (at != resources.end() && at->SourceEntity == source) return at->CopyEntity;
        const auto copy = create();
        if (copy != source) resources.insert(at, {source, copy});
        return copy;
    };
    for (const auto &pair : copies) r.remove<CopyOf>(pair.CopyEntity);
    std::vector<SourceCopy> geometries, armatures;
    for (const auto &pair : copies) {
        [&]<typename... Ts>(TypeList<Ts...>) {
            ([&] {
                if (const auto *v = r.try_get<const Ts>(SourceOfCopy(r, pair.SourceEntity))) r.emplace_or_replace<Ts>(pair.CopyEntity, Ts{*v});
            }(),
             ...);
        }(Shared{});
        if (linked) continue;
        if (const auto *geometry = r.try_get<const ObjectGeometry>(pair.CopyEntity)) {
            const auto mesh = copy_resource(geometries, geometry->Entity, [&] { return GraphRecord(r, geometry->Entity) != InvalidOffset ? action::nodes::CopyGraph(r, geometry->Entity, *clone++) : geometry->Entity; });
            if (mesh != geometry->Entity) r.replace<ObjectGeometry>(pair.CopyEntity, mesh);
        }
        if (const auto *arm = r.try_get<const ArmatureObject>(pair.CopyEntity)) {
            const auto data = arm->Entity;
            const auto copy = copy_resource(armatures, data, [&] {
                const auto copy = r.create();
                r.emplace<Armature>(copy, Armature{r.get<const Armature>(data)});
                return copy;
            });
            r.patch<ArmatureObject>(pair.CopyEntity, [&](auto &v) { v.Entity = copy; });
        }
    }
    for (const auto &pair : copies) {
        if (const auto *collider = r.try_get<const ColliderShape>(pair.CopyEntity); collider && CopyFor(geometries, collider->MeshEntity) != collider->MeshEntity) r.patch<ColliderShape>(pair.CopyEntity, [&](auto &v) { v.MeshEntity = CopyFor(geometries, v.MeshEntity); });
    }
    // A copied graph refers to the copy of each object it referred to.
    for (const auto &[_, copied] : geometries) {
        if (r.all_of<graph::GeometryGraph>(copied)) MapReferences(r, copied, [&](state::Entity object) { return CopyFor(copies, object); });
    }
    for (const auto &pair : copies) {
        if (const auto *modifier = r.try_get<const ArmatureModifier>(pair.CopyEntity); modifier && CopyFor(armatures, modifier->ArmatureEntity) != modifier->ArmatureEntity) r.patch<ArmatureModifier>(pair.CopyEntity, [&](auto &v) { v.ArmatureEntity = CopyFor(armatures, v.ArmatureEntity); });
        if (const auto *attachment = r.try_get<const BoneAttachment>(pair.CopyEntity); attachment && CopyFor(armatures, attachment->ArmatureEntity) != attachment->ArmatureEntity) r.patch<BoneAttachment>(pair.CopyEntity, [&](auto &v) { v.ArmatureEntity = CopyFor(armatures, v.ArmatureEntity); });
        if (const auto *name = r.try_get<const Name>(pair.CopyEntity)) {
            const std::string stem{NameStem(name->Value)};
            r.remove<Name>(pair.CopyEntity);
            EmplaceUniqueName(r, pair.CopyEntity, stem);
        }
    }
}
// The graph whose page shows the object: the subgraph that holds it, or else the scene of its root.
state::Entity GraphShowing(const state::Registry &r, state::Entity object) {
    if (const auto holder = SubgraphHolding(r, object); holder != state::Null) return holder;
    const auto *membership = RootMembership(r, object);
    return membership ? membership->Scenes.front() : FrameScene(r);
}
} // namespace

state::Entity MakeInstanceUnique(state::Registry &r, state::Entity instance) {
    const auto placed = [&] {
        if (!r.valid(instance) || !r.all_of<Instance>(instance) || r.all_of<CopyOf>(instance)) return false;
        const auto *producer = r.try_get<const ObjectGeometry>(ParentOrNull(r, instance));
        const auto *generated = producer ? r.try_get<const graph::GraphInstances>(producer->Entity) : nullptr;
        return !generated || !std::ranges::contains(generated->Entities, instance);
    };
    if (!placed()) {
        action::Fail(r, "Choose an object with an Instance, placed in a scene");
        return state::Null;
    }
    const auto copies = r.get<const Instance>(instance).Copies;
    const auto source = r.get<const Instance>(instance).SubgraphEntity;
    // The copies of the subgraph's own entities become objects of their own.
    auto own = copies;
    std::erase_if(own, [&](const auto &pair) { return r.all_of<CopyOf>(pair.SourceEntity); });
    // The copy of an object with an Instance has an Instance of the same subgraph with the same port values.
    // The copies of that Instance's copies are its copies.
    for (const auto &pair : own) {
        const auto *nested = r.try_get<const Instance>(pair.SourceEntity);
        if (!nested) continue;
        auto nested_copies = nested->Copies;
        for (auto &inner : nested_copies) {
            inner.CopyEntity = CopyFor(copies, inner.CopyEntity);
            r.replace<CopyOf>(inner.CopyEntity, pair.CopyEntity, inner.SourceEntity);
        }
        r.emplace<Instance>(pair.CopyEntity, nested->SubgraphEntity, nested->PortValues, std::move(nested_copies));
    }
    std::vector<uint32_t> records;
    AppendGraphRecords(r, own, records);
    const auto clones = CloneRecords(r, records);
    auto clone = clones.begin();
    Detach(r, own, clone, false);
    r.remove<Instance>(instance);
    // The copies of the subgraph's objects are the items of a new subgraph, whose ports promote the copies' fields.
    const auto objects = SubgraphObjects(r, source);
    std::vector<state::Entity> items;
    for (const auto &pair : own) {
        if (std::ranges::binary_search(objects, pair.SourceEntity)) items.push_back(pair.CopyEntity);
    }
    std::ranges::sort(items);
    auto interface = r.get<const graph::SubgraphInterface>(source);
    std::erase_if(interface.Ports, [&](auto &port) {
        auto *field = std::get_if<graph::PromotedField>(&port.Binding);
        if (!field) return true;
        const auto copy = CopyFor(copies, field->Entity);
        return copy == std::exchange(field->Entity, copy);
    });
    return graph::CreateSubgraphOfItems(r, GraphShowing(r, instance), std::move(items), std::move(interface), NameStem(GetName(r, source)));
}

std::vector<std::vector<state::Entity>> DuplicateObjects(state::Registry &r, std::span<const std::vector<state::Entity>> groups, bool linked) {
    // A holder stands in for an instance's object while the copies are made.
    std::vector<state::Entity> holders;
    std::vector<std::vector<SourceCopy>> held;
    std::vector<uint32_t> records;
    for (const auto &group : groups) {
        // An object with an Instance duplicates with an Instance of the same subgraph, which makes its copies.
        auto closure = ObjectsWithParts(r, group);
        const auto instances = closure | std::views::filter([&](state::Entity e) { return r.all_of<Instance>(e); }) | std::ranges::to<std::vector>();
        std::erase_if(closure, [&](state::Entity e) {
            const auto *copy = r.try_get<const CopyOf>(e);
            return copy && std::ranges::binary_search(instances, copy->ObjectEntity);
        });
        PublishNow(r, holders.emplace_back(r.create()), state::Null, held.emplace_back(), closure);
        if (!linked) AppendGraphRecords(r, held.back(), records);
    }
    const auto clones = CloneRecords(r, records);
    auto clone = clones.begin();
    std::vector<std::vector<state::Entity>> duplicates;
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto &copies = held[i];
        Detach(r, copies, clone, linked);
        for (const auto &pair : copies) {
            const auto *instance = r.try_get<const Instance>(pair.SourceEntity);
            if (instance) Attach(r, pair.CopyEntity, instance->SubgraphEntity, instance->PortValues);
        }
        // The root's duplicate takes the root's place in the world.
        const auto root = groups[i].front();
        const auto duplicate = CopyFor(copies, root);
        r.replace<Transform>(duplicate, *WorldTransformOf(r, root));
        r.emplace<SceneMembership>(duplicate, std::vector{FrameScene(r)});
        duplicates.push_back(groups[i] | std::views::transform([&](state::Entity e) { return CopyFor(copies, e); }) | std::ranges::to<std::vector>());
        r.destroy(holders[i]);
    }
    return duplicates;
}
