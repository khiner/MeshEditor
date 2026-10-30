#include "object/PendingSync.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/HierarchyInstance.h"
#include "state/EntityReferences.h"

namespace {
using namespace procedure;
using Member = HierarchyInstance::Member;
using Change = HierarchyChanges::Change;
// Only procedures whose object inputs differ need their own evaluated publication.
bool Contextual(const state::Registry &r, state::Entity mesh, std::span<const Member> members) {
    const auto *procedure = r.try_get<const GeometryProcedure>(mesh);
    if (!procedure) return false;
    for (auto node = procedure->First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) {
        bool contextual = false;
        VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
            if constexpr (ObjectInput<K> != nullptr) {
                const auto object = r.get<const K>(node).*ObjectInput<K>;
                contextual = MapHierarchyMember(members, object) != object;
            }
        });
        if (contextual) return true;
    }
    return false;
}
void Sync(state::Registry &r, state::Entity mesh, std::span<const Member> members, std::span<const Change> changes) {
    auto binding = r.get<const HierarchyGeometry>(mesh);
    const auto &source = r.get<const GeometryProcedure>(binding.Source);
    const auto authored = NodesOf(r, source);
    const auto removed = ReconcileHierarchyMembers(r, binding.Nodes, authored);
    const bool structural = binding.Nodes != r.get<const HierarchyGeometry>(mesh).Nodes || std::ranges::any_of(changes, [&](const auto &c) {
                                return c.Entity == binding.Source || (std::ranges::contains(authored, c.Entity) && (c.Type == state::Slot<ProcedureNode> || c.Type == state::Slot<NodeInputs> || c.Type == state::Slot<SubnetInterface> || c.Type == state::Slot<NodeContainer>));
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
        if (fresh) r.emplace<NodeLayout>(node.Entity, r.get<const NodeLayout>(node.Source));
        if (!structural) continue;
        r.emplace_or_replace<ProcedureNode>(node.Entity, mesh, map(r.get<const ProcedureNode>(node.Source).Next), kind);
        copy.template operator()<NodeInputs>(true);
        copy.template operator()<NodeContainer>(true);
        copy.template operator()<SubnetInterface>(true);
    }
    if (structural) {
        r.emplace_or_replace<GeometryProcedure>(mesh, map(source.First), map(source.Last), map(source.DisplayNode), source.DisplayPort);
        r.replace<HierarchyGeometry>(mesh, std::move(binding));
        r.destroy(removed);
    }
}
} // namespace

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
