#pragma once

#include "Transform.h"
#include "entt_fwd.h"
#include "numeric/mat4.h"

struct WorldTransform;

// Defined in Scene.cpp
void UpdateModelBuffer(entt::registry &, entt::entity, const WorldTransform &);

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

mat4 ToMatrix(const WorldTransform &);

Transform GetTransform(const entt::registry &, entt::entity);
void UpdateWorldTransform(entt::registry &, entt::entity, bool propagate_to_children = true);
entt::entity GetParentEntity(const entt::registry &, entt::entity);
void ClearParent(entt::registry &, entt::entity child);
void SetParent(entt::registry &, entt::entity child, entt::entity parent);
