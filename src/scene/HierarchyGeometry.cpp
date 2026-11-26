#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/HierarchyInstance.h"
#include "state/EntityReferences.h"

namespace {
using namespace procedure;
using Member = HierarchyInstance::Member;
using Change = HierarchyChanges::Change;
// Only procedures whose object inputs differ need their own evaluated publication.
bool Contextual(const state::Registry &r, state::Entity mesh, std::span<const Member> members) { return r.all_of<GeometryProcedure>(mesh) && MapsHierarchyReference(r, mesh, members); }
void Sync(state::Registry &r, state::Entity mesh, std::span<const Member> members, std::span<const Change> changes) {
    auto binding = r.get<const HierarchyGeometry>(mesh);
    const auto &source = r.get<const GeometryProcedure>(binding.Source);
    const auto authored = NodesOf(r, binding.Source);
    const auto removed = ReconcileHierarchyMembers(r, binding.Nodes, authored);
    const bool structural = binding.Nodes != r.get<const HierarchyGeometry>(mesh).Nodes || std::ranges::any_of(changes, [&](const auto &c) {
                                return c.Entity == binding.Source || (std::ranges::contains(authored, c.Entity) && (c.Type == state::Slot<ProcedureNode> || c.Type == state::Slot<NodeInputs> || c.Type == state::Slot<Subnet>));
                            });
    const auto map = [&](state::Entity e) { return MapHierarchyMember(binding.Nodes, e); };
    for (const auto &node : binding.Nodes) {
        const auto kind = r.get<const ProcedureNode>(node.Source).Kind;
        const bool fresh = !r.all_of<ProcedureNode>(node.Entity);
        const auto copy = [&]<typename T>(bool structure = false) {
            bool changed = fresh || structure || std::ranges::binary_search(changes, Change{node.Source, state::Slot<T>});
            if constexpr (ObjectInput<T> != nullptr) changed |= !r.all_of<T>(node.Entity) || r.get<const T>(node.Entity).*ObjectInput<T> != MapHierarchyMember(members, r.get<const T>(node.Source).*ObjectInput<T>);
            if (!changed) return;
            if (const auto *value = r.try_get<const T>(node.Source)) {
                auto copied = *value;
                state::ForEachEntity(copied, [&](auto &e) { e = MapHierarchyMember(members, map(e)); });
                r.emplace_or_replace<T>(node.Entity, std::move(copied));
            } else r.remove<T>(node.Entity);
        };
        VisitKind(kind, copy);
        copy.template operator()<InlineSelection>();
        copy.template operator()<SourceState>();
        copy.template operator()<NodeBypassed>();
        if (!structural) continue;
        r.emplace_or_replace<ProcedureNode>(node.Entity, mesh, map(r.get<const ProcedureNode>(node.Source).Next), kind);
        copy.template operator()<NodeInputs>(true);
        copy.template operator()<Subnet>(true);
    }
    if (structural) {
        // A display inside a subnet node keeps its path, whose first subnet node is the copy's.
        auto path = source.DisplayPath;
        if (!path.empty()) path.front() = map(path.front());
        const auto &ends = r.get<const NetworkNodes>(binding.Source);
        r.emplace_or_replace<NetworkNodes>(mesh, map(ends.First), map(ends.Last));
        r.emplace_or_replace<GeometryProcedure>(mesh, path.empty() ? map(source.DisplayNode) : source.DisplayNode, source.DisplayPort, std::move(path));
        r.replace<HierarchyGeometry>(mesh, std::move(binding));
        r.destroy(removed);
    }
}
} // namespace

bool MapsHierarchyReference(const state::Registry &r, state::Entity network, std::span<const Member> members) {
    for (const auto node : NodesOf(r, network)) {
        bool maps = false;
        VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
            if constexpr (ObjectInput<K> != nullptr) {
                const auto object = r.get<const K>(node).*ObjectInput<K>;
                maps = MapHierarchyMember(members, object) != object;
            } else if constexpr (std::same_as<K, Subnet>) {
                const auto definition = r.get<const Subnet>(node).DefinitionEntity;
                maps = definition != state::Null && MapsHierarchyReference(r, definition, members);
            }
        });
        if (maps) return true;
    }
    return false;
}

state::Entity ContextObject(const state::Registry &r, state::Entity mesh, state::Entity object) {
    const auto *context = r.try_get<const HierarchyGeometry>(mesh);
    return context ? MapHierarchyMember(r.get<const HierarchyInstance>(context->Root).Members, ContextObject(r, context->Source, object)) : object;
}

void PublishHierarchyGeometry(state::Registry &r, state::Entity root, std::span<const HierarchyChanges::Change> changes) {
    using namespace procedure;
    const auto &members = r.get<const HierarchyInstance>(root).Members;
    std::vector<HierarchyInstance::Member> meshes;
    for (const auto &member : members) {
        const auto *instance = r.try_get<const Instance>(member.Source);
        if (!instance || !Contextual(r, instance->Entity, members)) continue;
        if (MapHierarchyMember(meshes, instance->Entity) != instance->Entity) continue;
        auto mesh = state::Null;
        for (const auto [e, binding] : r.view<const HierarchyGeometry>().each())
            if (binding.Root == root && binding.Source == instance->Entity) {
                mesh = e;
                break;
            }
        if (mesh == state::Null) {
            mesh = r.create();
            r.emplace<HierarchyGeometry>(mesh, root, instance->Entity);
            r.emplace<GeometryProcedure>(mesh);
            r.emplace<NetworkNodes>(mesh);
            r.emplace<NetworkActive>(mesh);
        }
        meshes.insert(std::ranges::lower_bound(meshes, instance->Entity, {}, &HierarchyInstance::Member::Source), {instance->Entity, mesh});
    }
    for (const auto &member : members) {
        if (const auto *instance = r.try_get<const Instance>(member.Source)) {
            const auto mapped = MapHierarchyMember(members, MapHierarchyMember(meshes, instance->Entity));
            const auto previous = r.get<const Instance>(member.Entity).Entity;
            if (previous != mapped) {
                r.replace<Instance>(member.Entity, mapped);
                if (r.all_of<HierarchyGeometry>(previous)) r.Resources.emplace<PendingObjectRemovals>().Buffers.emplace(previous);
            }
        }
        if (const auto *collider = r.try_get<const ColliderShape>(member.Source)) {
            const auto mapped = MapHierarchyMember(members, MapHierarchyMember(meshes, collider->MeshEntity));
            if (r.get<const ColliderShape>(member.Entity).MeshEntity != mapped) r.patch<ColliderShape>(member.Entity, [&](auto &c) { c.MeshEntity = mapped; });
        }
    }
    for (const auto &mesh : meshes) Sync(r, mesh.Entity, members, changes);
}
