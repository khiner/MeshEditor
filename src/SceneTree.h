#pragma once

#include "numeric/mat4.h"

#include <entt/entity/entity.hpp>

struct SceneNode {
    entt::entity Parent{entt::null};
    entt::entity FirstChild{entt::null};
    entt::entity NextSibling{entt::null};
};

struct WorldMatrix {
    WorldMatrix(mat4 m)
        : M{std::move(m)}, MInv{glm::transpose(glm::inverse(M))} {}

    mat4 M; // World-space matrix
    mat4 MInv; // Transpose of inverse
};

// Inverse of parent's world matrix at the moment of parenting.
// Used to compute world matrices that follow parent transforms.
// WorldMatrix = ParentWorldMatrix * ParentInverse * LocalMatrix
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
    ChildrenIterator operator++(int);
    bool operator==(const ChildrenIterator &) const = default;
};

struct Children {
    const entt::registry *R;
    entt::entity ParentEntity;

    ChildrenIterator begin() const;
    ChildrenIterator end() const { return {R, entt::null}; }
};

entt::entity GetRootEntity(const entt::registry &, entt::entity);
entt::entity GetParentEntity(const entt::registry &, entt::entity); // If no parent, returns the provided entity.

void SetParent(entt::registry &, entt::entity child, entt::entity parent);
void ClearParent(entt::registry &, entt::entity child);
