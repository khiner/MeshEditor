#pragma once

#include "numeric/mat4.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"

#include <span>

// A node's authored parent, present only on a node with a parent.
struct SceneParent {
    state::Entity Parent;
};

// A node's links in the child lists: the head of its own children and the next child of its parent.
struct SceneChildren {
    state::Entity FirstChild{state::Null};
    state::Entity NextSibling{state::Null};
};

// Iterates a node's children.
struct ChildrenIterator {
    using difference_type = std::ptrdiff_t;
    using value_type = state::Entity;

    const state::Scene *R;
    state::Entity Current;

    state::Entity operator*() const { return Current; }
    ChildrenIterator &operator++();
    ChildrenIterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }
    bool operator==(const ChildrenIterator &) const = default;
};

struct Children {
    const state::Scene *R;
    state::Entity ParentEntity;

    ChildrenIterator begin() const;
    ChildrenIterator end() const { return {R, state::Null}; }
};

mat4 GetParentDelta(const state::Scene &, state::Entity);

// The node's parent, or null at a root.
state::Entity ParentOrNull(const state::Scene &, state::Entity);

// The nearest of `e` and its ancestors that `pred` matches, or state::Null when none does.
state::Entity FindAncestorIf(const state::Scene &r, state::Entity e, auto &&pred) {
    for (; e != state::Null && !pred(e); e = ParentOrNull(r, e)) {}
    return e;
}

// The local transform composing into the world transform: the pose when present, else the authored Transform.
const Transform *ComposedLocal(const state::Scene &, state::Entity);
// Writes an edited local transform.
// A posed node takes the edit in its pose, and components the active animation does not pose persist in its Transform.
void CommitEditedLocal(state::Scene &, state::Entity, const Transform &edited);
void PatchEditedLocal(state::Scene &r, state::Entity e, auto &&fn) {
    const auto *current = ComposedLocal(r, e);
    if (!current) return;
    Transform edited = *current;
    fn(edited);
    CommitEditedLocal(r, e, edited);
}

// The node's world transform, composed from its local transform under its parent's while the settle pass has not composed it yet.
Transform ComposeWorldTransform(const state::Scene &, state::Entity);

// Recomputes the world transforms of `roots` and their descendants top-down, writing each one whose value changed or whose instance slot was placed this settle.
// A node whose world transform holds its value leaves its subtree as it is, so every node whose local transform, parent or instance slot changed is a root.
// Nodes in `owned` keep the world transform another system writes, and the recompute leaves them and their subtrees alone.
// Nodes in `held` recompute while their descendants hold still.
// `owned` and `held` are in entity index order.
void RecomputeWorldTransforms(state::Scene &, std::span<const state::Entity> roots, std::span<const state::Entity> owned, std::span<const state::Entity> held);
