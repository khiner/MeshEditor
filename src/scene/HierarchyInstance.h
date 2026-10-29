#pragma once
#include <span>

#include "gpu/Transform.h"
#include "scene/SceneSchema.h"
#include "state/DirtySet.h"
#include "state/Registry.h"

struct HierarchyInstance {
    struct Member {
        state::Entity Source, Entity;
        auto operator<=>(const Member &) const = default;
    };
    state::Entity Source;
    std::vector<Member> Members;
};
struct HierarchyGeometry {
    state::Entity Root, Source;
    std::vector<HierarchyInstance::Member> Nodes;
};
struct HierarchyMember {
    state::Entity Root, Source;
};
// A source whose scene placement was deleted remains available to its occurrences.
struct HierarchyDefinition {};
struct HierarchyChanges {
    struct Change {
        state::Entity Entity;
        state::TypeId Type;
        auto operator<=>(const Change &) const = default;
    };
    std::vector<Change> Values;
    bool Publishing{};
};
void Reset(HierarchyChanges &, state::Registry &);
void RegisterHierarchyInstances(state::Registry &);
void PublishHierarchyInstances(state::Registry &);
void CollectHierarchyDefinitions(state::Registry &);
std::vector<state::Entity> HierarchyClosure(const state::Registry &, state::Entity);
std::vector<state::Entity> ReconcileHierarchyMembers(state::Registry &, std::vector<HierarchyInstance::Member> &, std::span<const state::Entity> sources, state::Entity source = state::Null, state::Entity root = state::Null);
state::Entity MapHierarchyMember(std::span<const HierarchyInstance::Member>, state::Entity);
std::pair<state::Entity, state::Entity> HierarchyReferenceTarget(const state::Registry &, state::Entity owner, state::Entity target, bool placement = false);
state::Entity HierarchySource(const state::Registry &, state::Entity);
bool HierarchyParentCycle(const state::Registry &, state::Entity child, state::Entity parent);
state::Entity CreateHierarchyInstance(state::Registry &, state::Entity source, const Transform &, state::Entity scene = state::Null);
void MakeHierarchyIndependent(state::Registry &, state::Entity);
// Expand occurrence removal and retain any referenced definition before ordinary object removal.
std::vector<state::Entity> PrepareHierarchyRemoval(state::Registry &, std::span<const state::Entity>);

void PublishHierarchyGeometry(state::Registry &, state::Entity root, std::span<const HierarchyChanges::Change>);
state::Entity HierarchyEditTarget(const state::Registry &, state::Entity, bool placement = false);
void BindHierarchyInstance(state::Registry &, state::Entity root, state::Entity source);

// Visit inherited root placements from the definition out to the occurrence.
void ForEachHierarchyPlacement(const state::Registry &r, state::Entity entity, auto &&visit) {
    if (const auto *member = r.try_get<const HierarchyMember>(entity)) {
        ForEachHierarchyPlacement(r, member->Source, visit);
        if (member->Root == entity) visit(entity, member->Source);
    }
}
