#pragma once

#include "Transform.h"
#include "numeric/mat4.h"

#include <entt_fwd.h>

struct SceneNode {
    entt::entity Parent{null_entity};
    entt::entity FirstChild{null_entity};
    entt::entity NextSibling{null_entity};
};

struct WorldMatrix {
    WorldMatrix(mat4 m)
        : M{std::move(m)}, MInv{glm::transpose(glm::inverse(M))} {}

    mat4 M; // World-space matrix
    mat4 MInv; // Transpose of inverse
};

void UpdateModelBuffer(entt::registry &, entt::entity, const WorldMatrix &); // actually defined in Scene.cpp

// Inverse of parent's world matrix at the moment of parenting.
// Used to compute world matrices that follow parent transforms.
// WorldMatrix = ParentWorldMatrix * ParentInverse * LocalMatrix
struct ParentInverse {
    mat4 M{I4};
};

struct Visible {};

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
    ChildrenIterator end() const { return {R, null_entity}; }
};

entt::entity GetRootEntity(const entt::registry &, entt::entity);
entt::entity GetParentEntity(const entt::registry &, entt::entity); // If no parent, returns the provided entity.

void SetParent(entt::registry &, entt::entity child, entt::entity parent);
void ClearParent(entt::registry &, entt::entity child);

Transform GetTransform(const entt::registry &, entt::entity);

// Recursively update world matrices of entity and its children based on current transforms
void UpdateWorldMatrix(entt::registry &, entt::entity);
