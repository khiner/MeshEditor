#pragma once
#include <span>

#include "gpu/Transform.h"
#include "graph/NumericValue.h"
#include "object/ObjectCreateInfo.h"
#include "scene/SceneSchema.h"
#include "state/DirtySet.h"
#include "state/Registry.h"

// One source entity of a scene subgraph and the entity that shows it in an Instance.
struct SourceCopy {
    state::Entity SourceEntity, CopyEntity;
    auto operator<=>(const SourceCopy &) const = default;
};
// The use of a scene subgraph. Persistent, always on an object of type Empty.
// The object shows a copy of each entity the subgraph holds, placed by the object's transform.
// A copy is an entity of its own, which the publish step keeps equal to its source and then sets the port values on.
// A copy shares what its source refers to: the geometry it draws, its materials and its armature.
struct Instance {
    state::Entity SubgraphEntity{state::Null};
    std::vector<graph::PortValue> PortValues; // Its own value of each port that has one, sorted by port id
    std::vector<SourceCopy> Copies; // The copy of each entity of the subgraph's closure, sorted by source
};
// Persistent on a copy: the entity it copies and the object whose Instance shows it.
struct CopyOf {
    state::Entity ObjectEntity, SourceEntity;
};
// Uses holds whether an Instance or a prototype changed since the last collection of unplaced subgraphs.
struct InstanceChanges {
    struct Change {
        state::Entity Entity;
        state::TypeId Type;
        auto operator<=>(const Change &) const = default;
    };
    std::vector<Change> Values;
    bool Publishing{}, Uses{};
};
void Reset(InstanceChanges &, state::Registry &);
void RegisterInstances(state::Registry &);
void PublishSubgraphInstances(state::Registry &);
// Destroys each unplaced subgraph that no Instance and no prototype uses, with its items.
void CollectUnplacedSubgraphs(state::Registry &);

// The objects the scene subgraph holds, through its nested subgraphs, sorted.
std::vector<state::Entity> SubgraphObjects(const state::Registry &, state::Entity subgraph);
// The entities an Instance of the scene subgraph copies, sorted: each object the subgraph holds with its parts.
// An object's parts are its descendants that are no items of a scene's graph, and its armature's joints.
std::vector<state::Entity> SubgraphClosure(const state::Registry &, state::Entity subgraph);
// The objects with their parts, sorted.
std::vector<state::Entity> ObjectsWithParts(const state::Registry &, std::vector<state::Entity> objects);
// The object with its descendants and their parts, sorted.
std::vector<state::Entity> ObjectClosure(const state::Registry &, state::Entity object);
// The subgraph that holds the item, or null for an item of a scene's or the document's own graph.
state::Entity SubgraphHolding(const state::Registry &, state::Entity item);
// The scene subgraphs the object uses: an Instance's subgraph, and the prototype of each Instance on Points node of the geometry it draws.
std::vector<state::Entity> UsedSubgraphs(const state::Registry &, state::Entity object);
// Whether an item of the holder that uses the subgraph would have an Instance copy itself.
// It would where the subgraph, through the subgraphs its items use at any depth, is the holder or a subgraph the holder is inside.
bool UseRecurses(const state::Registry &, state::Entity used, state::Entity holder);
// Whether an Instance of the subgraph copies the object, at any depth of the subgraphs its copies use.
bool SubgraphCopies(const state::Registry &, state::Entity subgraph, state::Entity object);
// Whether an Instance or a prototype uses the scene subgraph.
bool SubgraphInstanced(const state::Registry &, state::Entity subgraph);
// Whether the entity is a subgraph of a scene's graph, or one the document keeps for its Instances.
bool IsSceneSubgraph(const state::Registry &, state::Entity);

// Gives each source a copy, drops the copies of other entities, and returns the dropped copies.
std::vector<state::Entity> ReconcileCopies(state::Registry &, std::vector<SourceCopy> &, std::span<const state::Entity> sources);
// The copy of the source among the copies, or the entity itself when it has none.
state::Entity CopyFor(std::span<const SourceCopy>, state::Entity source);
// The Instance's own value of the port, or null.
const graph::NumericValue *PortValueOf(const Instance &, uint16_t port_id);
// Sets the own value of the port on the object's Instance.
void SetInstancePortValue(state::Registry &, state::Entity object, uint16_t port_id, graph::NumericValue);
// The entity a copy stands for, through copies of copies, or the entity itself.
state::Entity SourceOfCopy(const state::Registry &, state::Entity);
// The source that takes an edit of a copy's reference, with the target mapped to its own source when the same Instance shows it.
std::pair<state::Entity, state::Entity> CopyReferenceTarget(const state::Registry &, state::Entity owner, state::Entity target);
// Where an edit of a component field of the entity lands.
// A copy's field that the subgraph promotes to a port sets that port's value on the Instance of InstanceEntity.
// Any other field of a copy edits its source, by the same rule.
struct FieldTarget {
    state::Entity Entity;
    state::Entity InstanceEntity{state::Null};
    uint16_t PortId{};
};
FieldTarget CopyFieldTarget(const state::Registry &, state::Entity, state::TypeKey component, uint16_t offset);
// Whether the parent is the child, is below it, or is at or below an object whose Instance copies the child.
bool ParentCycle(const state::Registry &, state::Entity child, state::Entity parent);

// Adds an Empty with an Instance of the scene subgraph and the port values, and returns it.
state::Entity AddInstance(state::Registry &, state::Entity subgraph, const ObjectCreateInfo &, std::vector<graph::PortValue> = {});
// Adds an object a graph generates below the parent, with an Instance of the scene subgraph, and returns it.
// It has no marker geometry and is no member of a scene.
state::Entity AddGeneratedInstance(state::Registry &, state::Entity parent, state::Entity subgraph, const Transform &placement);
// Makes the object's Instance show another scene subgraph, with new copies and no port values.
void SetInstanceSubgraph(state::Registry &, state::Entity object, state::Entity subgraph);
// Makes the copies of the object's Instance the items of a new subgraph of the object's scene, removes the Instance, and returns the subgraph.
// Each copy takes its own copy of the resources it shared with its source.
// The copy of an object with an Instance has an Instance of the same subgraph, whose copies keep sharing.
// It reports a failure and returns null for an object that has no Instance or is not placed in a scene.
state::Entity MakeInstanceUnique(state::Registry &, state::Entity object);
// A duplicate of each group's objects with their parts, as independent objects, in each group's order.
// A group's first object is its root, whose duplicate stands at the root's world transform, and each other object is below an object of the group, whose duplicate its own hangs below.
// A linked duplicate draws its source's geometry and poses its source's armature, and any other duplicate has its own.
std::vector<std::vector<state::Entity>> DuplicateObjects(state::Registry &, std::span<const std::vector<state::Entity>> groups, bool linked);
// Moves a scene subgraph to the document: its items leave their scenes and stay for the Instances that copy them.
void UnplaceSubgraph(state::Registry &, state::Entity subgraph);
// The entities with the copies of each one's Instance, which ordinary object removal then removes.
std::vector<state::Entity> PrepareInstanceRemoval(state::Registry &, std::span<const state::Entity>);
