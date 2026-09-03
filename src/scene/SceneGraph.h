#pragma once

#include "entt_fwd.h"
#include "numeric/mat4.h"

struct SceneNode {
    entt::entity Parent{null_entity};
    entt::entity FirstChild{null_entity};
    entt::entity NextSibling{null_entity};
};

// Stores the Blender-style parent inverse used by WorldTransform = decompose(ParentMatrix * ParentInverse * LocalMatrix).
// Current callers initialize it to identity.
struct ParentInverse {
    mat4 M{I4};
};

// Iterator for traversing children of a SceneNode
struct ChildrenIterator {
    using difference_type = std::ptrdiff_t;
    using value_type = entt::entity;

    const entt::registry *R;
    entt::entity Current;

    entt::entity operator*() const { return Current; }
    ChildrenIterator &operator++();
    ChildrenIterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }
    bool operator==(const ChildrenIterator &) const = default;
};

struct Children {
    const entt::registry *R;
    entt::entity ParentEntity;

    ChildrenIterator begin() const;
    ChildrenIterator end() const { return {R, null_entity}; }
};

mat4 GetParentDelta(const entt::registry &, entt::entity);
entt::entity GetParentEntity(const entt::registry &, entt::entity);

// The node's parent, or null at a root.
entt::entity ParentOrNull(const entt::registry &, entt::entity);

// The nearest of `e` and its ancestors that `pred` matches, or null_entity when none does.
entt::entity FindAncestorIf(const entt::registry &r, entt::entity e, auto &&pred) {
    for (; e != null_entity && !pred(e); e = ParentOrNull(r, e)) {}
    return e;
}

// Build WorldTransform for `e`, and any ancestor still missing one, from local Transforms.
void EnsureWorldTransform(entt::registry &, entt::entity);

// Build WorldTransform for any entity that has none yet.
void BuildMissingWorldTransforms(entt::registry &);
