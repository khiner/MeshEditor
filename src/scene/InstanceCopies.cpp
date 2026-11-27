#include "SortUnique.h"
#include "armature/ArmatureComponents.h"
#include "graph/Graph.h"
#include "render/ObjectGeometry.h"
#include "scene/Entity.h"
#include "scene/Instance.h"
#include "scene/SceneGraph.h"

namespace {
// A part belongs to the object above it: a sub-element, a bone or its joints, or a copy an instance shows.
bool IsPart(const state::Registry &r, state::Entity e) {
    return !r.all_of<ObjectKind>(e) || r.any_of<SubElementOf, BoneSubPartOf, BoneJoint, CopyOf>(e);
}
// Appends what each entity from `first` brings with it: its parts, or with `descendants` each of its children, and its armature's joints.
// The instances a graph generates below its object stay out, since the copy's own graph generates them.
void Close(const state::Registry &r, std::vector<state::Entity> &closure, size_t first, bool descendants) {
    for (size_t i = first; i < closure.size(); ++i) {
        const auto e = closure[i];
        if (!r.valid(e)) continue;
        const auto *object = r.try_get<const ObjectGeometry>(e);
        const auto *generated = object ? r.try_get<const graph::GraphInstances>(object->Entity) : nullptr;
        for (const auto child : Children{&r, e}) {
            if (generated && std::ranges::contains(generated->Entities, child)) continue;
            if (descendants || IsPart(r, child)) closure.push_back(child);
        }
        if (const auto *arm = r.try_get<const ArmatureObject>(e)) {
            if (arm->JointEntity != state::Null) closure.push_back(arm->JointEntity);
        }
        if (const auto *joints = r.try_get<const BoneJointEntities>(e)) {
            closure.push_back(joints->Head);
            closure.push_back(joints->Tail);
        }
    }
    std::erase_if(closure, [&](auto e) { return !r.valid(e); });
    SortUnique(closure);
}
} // namespace

state::Entity CopyFor(std::span<const SourceCopy> copies, state::Entity source) {
    const auto at = std::ranges::lower_bound(copies, source, {}, &SourceCopy::SourceEntity);
    return at != copies.end() && at->SourceEntity == source ? at->CopyEntity : source;
}

std::vector<state::Entity> ReconcileCopies(state::Registry &r, std::vector<SourceCopy> &copies, std::span<const state::Entity> sources) {
    std::vector<state::Entity> removed;
    const auto existing = copies.size();
    copies.reserve(std::max(existing, sources.size()));
    for (const auto e : sources) {
        if (!std::ranges::binary_search(std::span{copies}.first(existing), e, {}, &SourceCopy::SourceEntity)) copies.push_back({e, r.create()});
    }
    if (copies.size() != existing) std::ranges::sort(copies, {}, &SourceCopy::SourceEntity);
    std::vector<state::Entity> sorted;
    if (!std::ranges::is_sorted(sources)) {
        sorted.assign(sources.begin(), sources.end());
        std::ranges::sort(sorted);
        sources = sorted;
    }
    std::erase_if(copies, [&](const auto &pair) {
        if (std::ranges::binary_search(sources, pair.SourceEntity)) return false;
        removed.push_back(pair.CopyEntity);
        return true;
    });
    return removed;
}

const graph::NumericValue *PortValueOf(const Instance &instance, uint16_t port_id) {
    const auto found = std::ranges::find(instance.PortValues, port_id, &graph::PortValue::PortId);
    return found == instance.PortValues.end() ? nullptr : &found->Value;
}

void SetInstancePortValue(state::Registry &r, state::Entity object, uint16_t port_id, graph::NumericValue value) {
    r.patch<Instance>(object, [&](Instance &held) {
        const auto found = std::ranges::lower_bound(held.PortValues, port_id, {}, &graph::PortValue::PortId);
        if (found == held.PortValues.end() || found->PortId != port_id) held.PortValues.insert(found, {port_id, value});
        else found->Value = value;
    });
}

state::Entity SourceOfCopy(const state::Registry &r, state::Entity entity) {
    while (const auto *copy = r.try_get<const CopyOf>(entity)) entity = copy->SourceEntity;
    return entity;
}

std::pair<state::Entity, state::Entity> CopyReferenceTarget(const state::Registry &r, state::Entity owner, state::Entity target) {
    const auto *copy = r.try_get<const CopyOf>(owner), *reference = r.try_get<const CopyOf>(target);
    if (copy && reference && copy->ObjectEntity == reference->ObjectEntity) target = SourceOfCopy(r, target);
    return {SourceOfCopy(r, owner), target};
}

FieldTarget CopyFieldTarget(const state::Registry &r, state::Entity entity, state::TypeKey component, uint16_t offset) {
    while (const auto *copy = r.try_get<const CopyOf>(entity)) {
        const auto subgraph = r.get<const Instance>(copy->ObjectEntity).SubgraphEntity;
        const graph::PortBinding field{graph::PromotedField{copy->SourceEntity, component, offset}};
        const auto &ports = r.get<const graph::SubgraphInterface>(subgraph).Ports;
        if (const auto port = std::ranges::find(ports, field, &graph::SubgraphPort::Binding); port != ports.end()) return {entity, copy->ObjectEntity, port->Id};
        entity = copy->SourceEntity;
    }
    return {entity};
}

std::vector<state::Entity> SubgraphObjects(const state::Registry &r, state::Entity subgraph) {
    std::vector<state::Entity> objects;
    const auto add = [&](this const auto &self, state::Entity held) -> void {
        for (const auto item : r.get<const graph::SubgraphItems>(held).Items) {
            if (!r.valid(item)) continue;
            if (r.all_of<graph::SubgraphItems>(item)) self(item);
            else if (r.all_of<ObjectKind>(item)) objects.push_back(item);
        }
    };
    add(subgraph);
    SortUnique(objects);
    return objects;
}

std::vector<state::Entity> ObjectsWithParts(const state::Registry &r, std::vector<state::Entity> objects) {
    Close(r, objects, 0u, false);
    return objects;
}

std::vector<state::Entity> SubgraphClosure(const state::Registry &r, state::Entity subgraph) { return ObjectsWithParts(r, SubgraphObjects(r, subgraph)); }

std::vector<state::Entity> ObjectClosure(const state::Registry &r, state::Entity object) {
    std::vector<state::Entity> closure{object};
    Close(r, closure, 0u, true);
    return closure;
}

state::Entity SubgraphHolding(const state::Registry &r, state::Entity item) {
    for (const auto [subgraph, held] : r.view<const graph::SubgraphItems>().each())
        if (std::ranges::binary_search(held.Items, item)) return subgraph;
    return state::Null;
}

std::vector<state::Entity> UsedSubgraphs(const state::Registry &r, state::Entity object) {
    std::vector<state::Entity> used;
    if (const auto *instance = r.try_get<const Instance>(object)) used.push_back(instance->SubgraphEntity);
    const auto *drawn = r.try_get<const ObjectGeometry>(object);
    const auto *nodes = drawn ? r.try_get<const graph::GraphNodes>(drawn->Entity) : nullptr;
    for (auto node = nodes ? nodes->First : state::Null; node != state::Null; node = r.get<const graph::GraphNode>(node).Next) {
        if (const auto *points = r.try_get<const graph::InstanceOnPoints>(node); points && points->Prototype != state::Null) used.push_back(points->Prototype);
    }
    return used;
}

namespace {
// Calls `found(subgraph, closure)` for the subgraph and for each subgraph that the entities an instance of it copies use, at any depth, until it returns true.
bool AnyUsed(const state::Registry &r, state::Entity subgraph, auto &&found) {
    std::vector<state::Entity> pending{subgraph}, visited;
    while (!pending.empty()) {
        const auto held = pending.back();
        pending.pop_back();
        if (!r.valid(held) || !r.all_of<graph::SubgraphItems>(held) || std::ranges::contains(visited, held)) continue;
        visited.push_back(held);
        const auto closure = SubgraphClosure(r, held);
        if (found(held, closure)) return true;
        for (const auto e : closure) pending.append_range(UsedSubgraphs(r, e));
    }
    return false;
}
} // namespace

bool UseRecurses(const state::Registry &r, state::Entity used, state::Entity holder) {
    std::vector<state::Entity> holders;
    for (; r.valid(holder) && r.all_of<graph::SubgraphItems>(holder); holder = r.get<const graph::SubgraphItems>(holder).ParentEntity) holders.push_back(holder);
    return !holders.empty() && AnyUsed(r, used, [&](state::Entity subgraph, const auto &) { return std::ranges::contains(holders, subgraph); });
}

bool SubgraphCopies(const state::Registry &r, state::Entity subgraph, state::Entity object) {
    return AnyUsed(r, subgraph, [&](state::Entity, const auto &closure) { return std::ranges::binary_search(closure, object); });
}

bool SubgraphInstanced(const state::Registry &r, state::Entity subgraph) {
    for (const auto [_, instance] : r.view<const Instance>().each())
        if (instance.SubgraphEntity == subgraph) return true;
    for (const auto [_, points] : r.view<const graph::InstanceOnPoints>().each())
        if (points.Prototype == subgraph) return true;
    return false;
}

bool ParentCycle(const state::Registry &r, state::Entity child, state::Entity parent) {
    for (auto e = parent; e != state::Null; e = ParentOrNull(r, e)) {
        if (e == child) return true;
        // An object stays out from under an object whose Instance shows it, where it would sit beside its own copy.
        if (const auto *instance = r.try_get<const Instance>(e); instance && SubgraphCopies(r, instance->SubgraphEntity, child)) return true;
    }
    return false;
}
