#pragma once

#include "numeric/mat4.h"
#include "state/Entity.h"

struct SceneNode {
    state::Entity Parent{null_entity};
    state::Entity FirstChild{null_entity};
    state::Entity NextSibling{null_entity};
};

// Stores the Blender-style parent inverse used by WorldTransform = decompose(ParentMatrix * ParentInverse * LocalMatrix).
// Current callers initialize it to identity.
struct ParentInverse {
    mat4 M{I4};
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
    ChildrenIterator end() const { return {R, null_entity}; }
};

mat4 GetParentDelta(const state::Scene &, state::Entity);
state::Entity GetParentEntity(const state::Scene &, state::Entity);

// The node's parent, or null at a root.
state::Entity ParentOrNull(const state::Scene &, state::Entity);

// The nearest of `e` and its ancestors that `pred` matches, or null_entity when none does.
state::Entity FindAncestorIf(const state::Scene &r, state::Entity e, auto &&pred) {
    for (; e != null_entity && !pred(e); e = ParentOrNull(r, e)) {}
    return e;
}

// Build WorldTransform for `e`, and any ancestor still missing one, from local Transforms.
void EnsureWorldTransform(state::Scene &, state::Entity);

// Build WorldTransform for any entity that has none yet.
void BuildMissingWorldTransforms(state::Scene &);
