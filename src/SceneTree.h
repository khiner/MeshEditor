#pragma once

#include "entt_fwd.h"
#include "gpu/Transform.h"
#include "numeric/mat4.h"

// Distinct component type so an entity can hold both local and world transforms.
struct WorldTransform : Transform {
    using Transform::Transform;
    WorldTransform(const Transform &t) : Transform{t} {}
};

struct SceneNode {
    entt::entity Parent{null_entity};
    entt::entity FirstChild{null_entity};
    entt::entity NextSibling{null_entity};
};

// Inverse of parent's world matrix at the moment of parenting.
// Used to compute world transforms that follow parent transforms.
// WorldTransform = decompose(ParentMatrix * ParentInverse * LocalMatrix)
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
void ClearParent(entt::registry &, entt::entity child);
void SetParent(entt::registry &, entt::entity child, entt::entity parent);
