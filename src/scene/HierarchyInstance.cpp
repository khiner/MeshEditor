#include "scene/HierarchyInstance.h"
#include "state/EntityReferences.h"

#include "Path.h"
#include "SortUnique.h"
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
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "procedure/Procedure.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/RenderChanges.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"

namespace {
using Values = TypeList<Transform, Name, Path, ObjectKind, Instance, Visibility, ScaleLocked, Perspective, Orthographic, PunctualLight, ArmatureObject, BoneJoint, BoneSubPartOf, BoneJointEntities, SubElementOf, ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, BoneDelta, BoneConstraints, PhysicsMotion, PhysicsVelocity, ColliderShape, ColliderMaterial, ColliderPolicy, TriggerTag, TriggerNodes, PhysicsJoint, Striker, ModalSoundControls, ContactSurface, SurfaceSoundControls, AcousticMaterial, SoundVerticesModel, ModalGain, ModalTuning, ModalSolveSettings, MassProperties, RealImpactMicrophone, RealImpactActiveMicrophone>;
using Definitions = TypeList<AnimationClips, VertexSamples, ModalModes, ModalEigenSummary, RealImpactVertices>;

using Member = HierarchyInstance::Member;

std::vector<state::Entity> ReferencedHierarchies(const state::Registry &r) {
    std::vector<state::Entity> sources;
    for (const auto [_, instance] : r.view<const HierarchyInstance>().each()) sources.push_back(instance.Source);
    for (const auto [_, points] : r.view<const procedure::InstanceOnPoints>().each()) sources.push_back(points.Prototype);
    SortUnique(sources);
    return sources;
}

template<typename T> void Changed(state::Registry &r, state::Entity e) {
    auto &changes = r.Resources.get<HierarchyChanges>();
    if (!changes.Publishing && !r.view<const HierarchyInstance>().empty()) changes.Values.emplace_back(e, state::Slot<T>);
}

template<typename T> void Copy(state::Registry &r, const Member &member, std::span<const Member> members) {
    if (const auto *value = r.try_get<const T>(member.Source)) {
        T copied = *value;
        state::ForEachEntity(copied, [&](auto &e) { e = MapHierarchyMember(members, e); });
        if constexpr (std::same_as<T, ArmatureObject>) {
            if (const auto *prior = r.try_get<const ArmatureObject>(member.Entity); prior && (prior->Entity != copied.Entity || prior->BoneEntities.size() != copied.BoneEntities.size())) r.remove<ArmaturePoseState>(member.Entity);
        }
        r.emplace_or_replace<T>(member.Entity, std::move(copied));
    } else r.remove<T>(member.Entity);
}

void CopyHelperGeometry(state::Registry &r, const Member &member) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (!r.all_of<VertexStoreId>(member.Source)) {
        r.remove<VertexStoreId, ModelsBuffer>(member.Entity);
        return;
    }
    if (r.all_of<VertexStoreId>(member.Entity)) return;
    if (r.all_of<ArmatureObject>(member.Source)) {
        const auto data = primitive::BoneOctahedron(1.f);
        r.emplace<VertexStoreId>(member.Entity, meshes.AllocateVertexBuffer(data.Mesh.Positions, data.Attrs));
    } else if (r.all_of<BoneJoint>(member.Source)) {
        const auto data = primitive::BoneSphereDisc();
        r.emplace<VertexStoreId>(member.Entity, meshes.AllocateVertexBuffer(data.Mesh.Positions, {}));
    }
}

void Publish(state::Registry &r, state::Entity root, std::span<const HierarchyChanges::Change> changes) {
    const auto source = r.get<const HierarchyInstance>(root).Source;
    const auto closure = HierarchyClosure(r, source);
    auto members = r.get<const HierarchyInstance>(root).Members;
    const auto removed = ReconcileHierarchyMembers(r, members, closure, source, root);
    const bool structure = members != r.get<const HierarchyInstance>(root).Members || std::ranges::any_of(changes, [&](const auto &c) {
                               return c.Type == state::Slot<SceneParent> && std::ranges::binary_search(closure, c.Entity);
                           });
    if (r.get<const HierarchyInstance>(root).Members != members) r.patch<HierarchyInstance>(root, [&](auto &instance) { instance.Members = members; });
    for (const auto &member : members) {
        const bool fresh = !r.all_of<HierarchyMember>(member.Entity);
        if (fresh) r.emplace<HierarchyMember>(member.Entity, root, member.Source);
        [&]<typename... Ts>(TypeList<Ts...>) {
            ([&] {
                if (member.Entity == root && (std::same_as<Ts, Transform> || std::same_as<Ts, Name> || std::same_as<Ts, Visibility> || std::same_as<Ts, SubElementOf>)) return;
                if (fresh || structure || std::ranges::binary_search(changes, HierarchyChanges::Change{member.Source, state::Slot<Ts>})) Copy<Ts>(r, member, members);
            }(),
             ...);
        }(Values{});
        CopyHelperGeometry(r, member);
        if (fresh || std::ranges::binary_search(changes, HierarchyChanges::Change{member.Source, state::Slot<MorphWeightRange>})) {
            const auto *weights = r.try_get<const MorphWeightRange>(member.Source);
            const auto *current = r.try_get<const MorphWeightRange>(member.Entity);
            auto &buffer = r.Resources.get<GpuBuffers>().MorphWeightBuffer;
            if (!weights) r.remove<MorphWeightRange>(member.Entity);
            else if (!current || current->Weights.Count != weights->Weights.Count) {
                r.remove<MorphWeightRange>(member.Entity);
                r.emplace<MorphWeightRange>(member.Entity, buffer.Clone(weights->Weights));
            } else {
                const auto source_values = buffer.Get(weights->Weights);
                if (!std::ranges::equal(source_values, buffer.Get(current->Weights))) std::ranges::copy(source_values, buffer.GetMutable(current->Weights).begin());
            }
            r.Resources.get<RenderChanges>().MorphWeights.emplace(member.Entity);
        }
    }
    if (structure) {
        std::vector<state::Entity> children;
        for (const auto &member : members)
            if (member.Entity != root) children.push_back(member.Entity);
        ClearParents(r, children);
        for (const auto &member : members) {
            if (member.Entity == root) continue;
            const auto parent = ParentOrNull(r, member.Source);
            if (parent != state::Null && MapHierarchyMember(members, parent) != parent) SetParent(r, member.Entity, MapHierarchyMember(members, parent));
            r.remove<SceneMembership>(member.Entity);
        }
    }
    Destroy(r, removed);
    PublishHierarchyGeometry(r, root, changes);
}
} // namespace

void Reset(HierarchyChanges &changes, state::Registry &) {
    changes.Values.clear();
    changes.Publishing = false;
}

state::Entity CreateHierarchyInstance(state::Registry &r, state::Entity source, const Transform &placement, state::Entity scene) {
    if (!r.all_of<ObjectKind>(source)) {
        action::Fail(r, "Choose an object hierarchy as the instance source");
        return state::Null;
    }
    const auto root = r.create();
    EmplaceObject(r, root, r.get<const ObjectKind>(source).Value, scene);
    r.emplace<Transform>(root, placement);
    r.emplace<Visibility>(root);
    EmplaceUniqueName(r, root, NameStem(GetName(r, source)));
    BindHierarchyInstance(r, root, source);
    return root;
}

void BindHierarchyInstance(state::Registry &r, state::Entity root, state::Entity source) {
    if (const auto *current = r.try_get<const HierarchyInstance>(root)) {
        if (current->Source == source) return;
        std::vector<state::Entity> removed;
        for (const auto &member : current->Members)
            if (member.Entity != root) removed.push_back(member.Entity);
        r.remove<HierarchyInstance, HierarchyMember>(root);
        Destroy(r, removed);
    }
    r.emplace<HierarchyInstance>(root, source);
    auto &changes = r.Resources.get<HierarchyChanges>();
    const bool publishing = std::exchange(changes.Publishing, true);
    Publish(r, root, {});
    changes.Publishing = publishing;
}

void PublishHierarchyInstances(state::Registry &r) {
    for (const auto [mesh, binding] : r.view<const HierarchyGeometry>().each()) {
        if (!r.all_of<HierarchyInstance>(binding.Root)) r.Resources.emplace<PendingObjectRemovals>().Buffers.emplace(mesh);
    }
    auto &changes = r.Resources.get<HierarchyChanges>();
    for (const auto e : r.Resources.get<const RenderChanges>().MorphWeights) {
        if (!r.all_of<HierarchyMember>(e)) changes.Values.emplace_back(e, state::Slot<MorphWeightRange>);
    }
    if (changes.Values.empty() || r.view<const HierarchyInstance>().empty()) {
        changes.Values.clear();
        return;
    }
    auto values = std::move(changes.Values);
    SortUnique(values);
    const auto roots = SortedEntities(r.view<const HierarchyInstance>());
    std::vector<state::Entity> ordered, visiting;
    const auto visit = [&](auto &&self, state::Entity root) -> void {
        if (std::ranges::contains(ordered, root) || std::ranges::contains(visiting, root)) return;
        visiting.push_back(root);
        for (const auto &member : r.get<const HierarchyInstance>(root).Members) {
            if (r.all_of<HierarchyInstance>(member.Source)) self(self, member.Source);
            else if (const auto *binding = r.try_get<const HierarchyMember>(member.Source); binding && binding->Root != root) self(self, binding->Root);
        }
        visiting.pop_back();
        ordered.push_back(root);
    };
    for (const auto root : roots) visit(visit, root);
    changes.Publishing = true;
    for (const auto root : ordered) {
        if (!r.all_of<HierarchyInstance>(root)) continue;
        const auto &instance = r.get<const HierarchyInstance>(root);
        const auto affected = [&](const auto &change) {
            if (change.Entity == instance.Source && (change.Type == state::Slot<Transform> || change.Type == state::Slot<Visibility> || change.Type == state::Slot<Name>)) return false;
            if (MapHierarchyMember(instance.Members, change.Entity) != change.Entity) return true;
            if (change.Type == state::Slot<SceneParent> && MapHierarchyMember(instance.Members, ParentOrNull(r, change.Entity)) != ParentOrNull(r, change.Entity)) return true;
            const auto *node = r.try_get<const procedure::ProcedureNode>(change.Entity);
            const auto mesh = node ? node->Procedure : r.all_of<procedure::GeometryProcedure>(change.Entity) ? change.Entity :
                                                                                                               state::Null;
            return mesh != state::Null && std::ranges::any_of(instance.Members, [&](const auto &member) {
                       const auto *placement = r.try_get<const Instance>(member.Source);
                       return placement && placement->Entity == mesh;
                   });
        };
        if (!std::ranges::any_of(values, affected)) continue;
        const auto before = instance.Members;
        Publish(r, root, values);
        // Published members may themselves belong to another hierarchy's definition.
        const auto count = values.size();
        size_t change = 0;
        for (const auto &m : r.get<const HierarchyInstance>(root).Members) {
            while (change < count && values[change].Entity < m.Source) ++change;
            for (; change < count && values[change].Entity == m.Source; ++change) values.push_back({m.Entity, values[change].Type});
            if (!std::ranges::binary_search(before, m.Source, {}, &Member::Source)) values.push_back({m.Entity, state::Slot<SceneParent>});
        }
        SortUnique(values);
    }
    changes.Publishing = false;
    changes.Values.clear();
}

void CollectHierarchyDefinitions(state::Registry &r) {
    while (!r.view<const HierarchyDefinition>().empty()) {
        const auto sources = ReferencedHierarchies(r);
        std::vector<state::Entity> unused;
        for (const auto definition : r.view<const HierarchyDefinition>()) {
            const auto closure = HierarchyClosure(r, definition);
            if (std::ranges::none_of(closure, [&](auto e) { return std::ranges::binary_search(sources, e); })) unused.append_range(closure);
        }
        if (unused.empty()) break;
        for (const auto e : unused) r.remove<HierarchyDefinition>(e);
        Destroy(r, unused);
    }
}

void RegisterHierarchyInstances(state::Registry &r) {
    r.Resources.emplace<HierarchyChanges>();
    const auto track = [&]<typename... Ts>(TypeList<Ts...>) {
        (r.on_construct<Ts, &Changed<Ts>>(), ...);
        (r.on_update<Ts, &Changed<Ts>>(), ...);
        (r.on_destroy<Ts, &Changed<Ts>>(), ...);
    };
    track(Values{});
    track(Definitions{});
    track(TypeList<SceneParent, MorphWeightRange, HierarchyInstance>{});
    track(procedure::NodeKinds{});
    track(TypeList<procedure::GeometryProcedure, procedure::ProcedureNode, procedure::NodeInputs, procedure::NodeContainer, procedure::SubnetInterface, procedure::NodeBypassed, procedure::InlineSelection, procedure::SourceState>{});
}

std::vector<state::Entity> PrepareHierarchyRemoval(state::Registry &r, std::span<const state::Entity> entities) {
    const auto sources = ReferencedHierarchies(r);
    std::vector<state::Entity> removed;
    for (const auto e : entities) {
        if (std::ranges::binary_search(sources, e)) {
            ClearParents(r, std::span{&e, 1u});
            r.remove<SceneMembership, Selected, Active>(e);
            r.emplace_or_replace<HierarchyDefinition>(e);
            Hide(r, e);
            continue;
        }
        if (const auto *instance = r.try_get<const HierarchyInstance>(e))
            for (const auto &member : instance->Members) removed.push_back(member.Entity);
        else removed.push_back(e);
    }
    SortUnique(removed);
    return removed;
}

void MakeHierarchyIndependent(state::Registry &r, state::Entity root) {
    const auto *instance = r.try_get<const HierarchyInstance>(root);
    if (!instance) {
        action::Fail(r, "Choose a hierarchy instance to make independent");
        return;
    }
    const auto &members = instance->Members;
    std::vector<Member> resources;
    const auto copy_resource = [&](state::Entity source, auto create) {
        const auto at = std::ranges::lower_bound(resources, source, {}, &Member::Source);
        if (at != resources.end() && at->Source == source) return at->Entity;
        const auto copy = create();
        if (copy != source) resources.insert(at, {source, copy});
        return copy;
    };
    for (const auto &member : members) {
        [&]<typename... Ts>(TypeList<Ts...>) {
            ([&] {
                if (const auto *v = r.try_get<const Ts>(HierarchySource(r, member.Source))) {
                    auto copied = *v;
                    if constexpr (std::same_as<Ts, AnimationClips>)
                        for (auto &clip : copied.Clips)
                            for (auto &channel : clip.Channels) animation::PlaceChannel(r, member.Entity, channel);
                    r.emplace_or_replace<Ts>(member.Entity, std::move(copied));
                }
            }(),
             ...);
        }(Definitions{});
        if (const auto *geometry = r.try_get<const Instance>(member.Entity)) {
            const auto mesh = copy_resource(geometry->Entity, [&] {
                const auto drawn = DrawnStoreId(r, geometry->Entity);
                return drawn && r.all_of<procedure::GeometryProcedure>(geometry->Entity) ? action::nodes::CopyProcedure(r, geometry->Entity, CloneRecord(r, *drawn)) : geometry->Entity;
            });
            if (mesh != geometry->Entity) r.replace<Instance>(member.Entity, mesh);
        }
        if (const auto *arm = r.try_get<const ArmatureObject>(member.Entity)) {
            const auto data = arm->Entity;
            const auto copy = copy_resource(data, [&] {
                const auto copy = r.create();
                r.emplace<Armature>(copy, Armature{r.get<const Armature>(data)});
                return copy;
            });
            r.patch<ArmatureObject>(member.Entity, [&](auto &v) { v.Entity = copy; });
        }
    }
    for (const auto &member : members) {
        if (r.all_of<ArmatureModifier>(member.Entity)) r.patch<ArmatureModifier>(member.Entity, [&](auto &v) { v.ArmatureEntity = MapHierarchyMember(resources, v.ArmatureEntity); });
        if (r.all_of<BoneAttachment>(member.Entity)) r.patch<BoneAttachment>(member.Entity, [&](auto &v) { v.ArmatureEntity = MapHierarchyMember(resources, v.ArmatureEntity); });
        if (r.all_of<ColliderShape>(member.Entity)) r.patch<ColliderShape>(member.Entity, [&](auto &v) { v.MeshEntity = MapHierarchyMember(resources, v.MeshEntity); });
        r.remove<HierarchyMember>(member.Entity);
    }
    for (const auto &[_, copied] : resources) {
        const auto *procedure = r.try_get<const procedure::GeometryProcedure>(copied);
        if (!procedure) continue;
        for (auto node = procedure->First; node != state::Null; node = r.get<const procedure::ProcedureNode>(node).Next) {
            procedure::VisitKind(r.get<const procedure::ProcedureNode>(node).Kind, [&]<typename K> {
                if constexpr (procedure::ObjectInput<K> != nullptr) r.patch<K>(node, [&](auto &p) { p.*procedure::ObjectInput<K> = MapHierarchyMember(members, p.*procedure::ObjectInput<K>); });
            });
        }
    }
    r.remove<HierarchyInstance>(root);
}
