#pragma once

#include "scene/WorldTransform.h"
#include "state/Registry.h"

struct SceneMembership;

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

    const state::Registry *R;
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
    const state::Registry *R;
    state::Entity ParentEntity;

    ChildrenIterator begin() const;
    ChildrenIterator end() const { return {R, state::Null}; }
};

mat4 GetParentDelta(const state::Registry &, state::Entity);

// The node's parent, or null at a root.
state::Entity ParentOrNull(const state::Registry &, state::Entity);

// The node's root holding its scenes, or null when its root has none.
// A sub-element without a parent resolves through the object it belongs to.
state::Entity MembershipRoot(const state::Registry &, state::Entity node);
// The scenes of the node's root, or null when its root has none.
const SceneMembership *RootMembership(const state::Registry &, state::Entity node);
// Whether the node's root is a member of the scene.
bool InScene(const state::Registry &, state::Entity node, state::Entity scene);

// The nearest of `e` and its ancestors that `pred` matches, or state::Null when none does.
state::Entity FindAncestorIf(const state::Registry &r, state::Entity e, auto &&pred) {
    for (; e != state::Null && !pred(e); e = ParentOrNull(r, e)) {}
    return e;
}

// The local transform composing into the world transform: the pose when present, else the authored Transform.
const Transform *ComposedLocal(const state::Registry &, state::Entity);
// Writes an edited local transform.
// A posed node takes the edit in its pose, and components the active animation does not pose persist in its Transform.
void CommitEditedLocal(state::Registry &, state::Entity, const Transform &edited);
void PatchEditedLocal(state::Registry &r, state::Entity e, auto &&fn) {
    const auto *current = ComposedLocal(r, e);
    if (!current) return;
    Transform edited = *current;
    fn(edited);
    CommitEditedLocal(r, e, edited);
}

// The node's transform relative to the frame, which is the node, an ancestor of it or null for world space, by the rule the settle pass composes world transforms with.
// An owned node's transform is its published world transform, so owned nodes lie only below a null frame.
// Every other node's local transform composes under its parent's transform.
// Sort owned by entity index.
Transform ComposeTransform(const state::Registry &, state::Entity, state::Entity frame, std::span<const state::Entity> owned);

// Recompute roots and descendants top-down.
// Include every node with a changed local transform, parent, or instance slot in roots.
// Skip owned nodes and subtrees; recompute held nodes while preserving descendants.
// Sort owned and held by entity index.
void RecomputeWorldTransforms(state::Registry &, std::span<const state::Entity> roots, std::span<const state::Entity> owned, std::span<const state::Entity> held);
