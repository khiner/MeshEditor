#include "SortUnique.h"
#include "armature/ArmatureComponents.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"

state::Entity MapHierarchyMember(std::span<const HierarchyInstance::Member> members, state::Entity entity) {
    const auto at = std::ranges::lower_bound(members, entity, {}, &HierarchyInstance::Member::Source);
    return at != members.end() && at->Source == entity ? at->Entity : entity;
}

std::vector<state::Entity> ReconcileHierarchyMembers(state::Registry &r, std::vector<HierarchyInstance::Member> &members, std::span<const state::Entity> sources, state::Entity source, state::Entity root) {
    std::vector<state::Entity> removed;
    const auto existing = members.size();
    members.reserve(std::max(existing, sources.size()));
    for (const auto e : sources) {
        if (!std::ranges::binary_search(std::span{members}.first(existing), e, {}, &HierarchyInstance::Member::Source)) members.push_back({e, e == source ? root : r.create()});
    }
    if (members.size() != existing) std::ranges::sort(members, {}, &HierarchyInstance::Member::Source);
    std::vector<state::Entity> sorted;
    if (!std::ranges::is_sorted(sources)) {
        sorted.assign(sources.begin(), sources.end());
        std::ranges::sort(sorted);
        sources = sorted;
    }
    std::erase_if(members, [&](const auto &member) {
        if (std::ranges::binary_search(sources, member.Source)) return false;
        removed.push_back(member.Entity);
        return true;
    });
    return removed;
}

std::pair<state::Entity, state::Entity> HierarchyReferenceTarget(const state::Registry &r, state::Entity owner, state::Entity target, bool placement) {
    const auto edited = HierarchyEditTarget(r, owner, placement);
    if (edited != owner) {
        const auto *member = r.try_get<const HierarchyMember>(owner), *reference = r.try_get<const HierarchyMember>(target);
        if (member && reference && member->Root == reference->Root) target = HierarchySource(r, target);
    }
    return {edited, target};
}

state::Entity HierarchySource(const state::Registry &r, state::Entity entity) {
    while (const auto *member = r.try_get<const HierarchyMember>(entity)) entity = member->Source;
    return entity;
}

state::Entity HierarchyEditTarget(const state::Registry &r, state::Entity entity, bool placement) {
    if (placement) {
        if (const auto *member = r.try_get<const HierarchyMember>(entity); member && member->Root == entity) return entity;
    }
    return HierarchySource(r, entity);
}

std::vector<state::Entity> HierarchyClosure(const state::Registry &r, state::Entity source) {
    std::vector<state::Entity> members{source};
    for (size_t i = 0; i < members.size(); ++i) {
        const auto e = members[i];
        if (!r.valid(e)) continue;
        const auto *object = r.try_get<const Instance>(e);
        const auto *generated = object ? r.try_get<const procedure::ProcedureInstances>(object->Entity) : nullptr;
        for (const auto child : Children{&r, e}) {
            if (!generated || !std::ranges::contains(generated->Entities, child)) members.push_back(child);
        }
        if (const auto *arm = r.try_get<const ArmatureObject>(e)) {
            if (arm->JointEntity != state::Null) members.push_back(arm->JointEntity);
        }
        if (const auto *joints = r.try_get<const BoneJointEntities>(e)) {
            members.push_back(joints->Head);
            members.push_back(joints->Tail);
        }
    }
    std::erase_if(members, [&](auto e) { return !r.valid(e); });
    SortUnique(members);
    return members;
}

bool HierarchyParentCycle(const state::Registry &r, state::Entity child, state::Entity parent) {
    std::vector<state::Entity> pending{child};
    std::vector<bool> visited(r.EntityCapacity());
    while (!pending.empty()) {
        const auto e = pending.back();
        pending.pop_back();
        if (e == parent) return true;
        if (!r.valid(e) || visited[state::Index(e)]) continue;
        visited[state::Index(e)] = true;
        for (const auto c : Children{&r, e}) pending.push_back(c);
        if (const auto *instance = r.try_get<const HierarchyInstance>(e)) pending.push_back(instance->Source);
        if (const auto *placement = r.try_get<const Instance>(e)) {
            if (const auto *procedure = r.try_get<const procedure::GeometryProcedure>(placement->Entity)) {
                for (auto node = procedure->First; node != state::Null; node = r.get<const procedure::ProcedureNode>(node).Next) {
                    if (const auto *points = r.try_get<const procedure::InstanceOnPoints>(node); points && points->Prototype != state::Null) pending.push_back(points->Prototype);
                }
            }
        }
    }
    return false;
}
