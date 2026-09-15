#pragma once

#include "numeric/mat4.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"

struct SceneNode {
    state::Entity Parent{state::Null};
    state::Entity FirstChild{state::Null};
    state::Entity NextSibling{state::Null};
};

// Iterator for traversing children of a SceneNode
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
state::Entity GetParentEntity(const state::Scene &, state::Entity);

// The node's parent, or null at a root.
state::Entity ParentOrNull(const state::Scene &, state::Entity);

// The nearest of `e` and its ancestors that `pred` matches, or state::Null when none does.
state::Entity FindAncestorIf(const state::Scene &r, state::Entity e, auto &&pred) {
    for (; e != state::Null && !pred(e); e = ParentOrNull(r, e)) {}
    return e;
}

// The local transform composing into WorldTransform: the pose when present, else the authored Transform.
const Transform *ComposedLocal(const state::Scene &, state::Entity);
// The local transform edits target: the authored Transform when present, else the pose.
const Transform *EditedLocal(const state::Scene &, state::Entity);
void PatchEditedLocal(state::Scene &r, state::Entity e, auto &&fn) {
    if (r.all_of<Transform>(e)) r.patch<Transform>(e, fn);
    else r.patch<PosedLocal>(e, [&](PosedLocal &posed) { fn(posed.Value); });
}

// Build WorldTransform for `e`, and any ancestor still missing one, from local transforms.
void EnsureWorldTransform(state::Scene &, state::Entity);

// Build WorldTransform for any entity that has none yet.
void BuildMissingWorldTransforms(state::Scene &);
