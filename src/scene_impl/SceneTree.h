#pragma once

#include "Transform.h"

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
    ChildrenIterator operator++(int);
    bool operator==(const ChildrenIterator &) const = default;
};

struct Children {
    const entt::registry *R;
    entt::entity ParentEntity;

    ChildrenIterator begin() const;
    ChildrenIterator end() const { return {R, null_entity}; }
};

Transform GetTransform(const entt::registry &r, entt::entity e) {
    return {r.get<Position>(e).Value, r.get<Rotation>(e).Value, r.all_of<Scale>(e) ? r.get<Scale>(e).Value : vec3{1}};
}

mat4 ToMatrix(const WorldTransform &wt) {
    return glm::translate(I4, wt.Position) * glm::mat4_cast(glm::normalize(Vec4ToQuat(wt.Rotation))) * glm::scale(I4, wt.Scale);
}

WorldTransform MakeWorldTransform(mat4 m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    glm::decompose(m, scale, rotation, translation, skew, perspective);
    return {translation, QuatToVec4(glm::normalize(rotation)), scale};
}

WorldTransform MakeWorldTransform(const Transform &t) {
    return {t.P, QuatToVec4(glm::normalize(t.R)), t.S};
}

// Recursively update world transforms of entity and its children based on current transforms
void UpdateWorldTransform(entt::registry &r, entt::entity e) {
    static const auto GetParentDelta = [](const entt::registry &r, entt::entity child) -> mat4 {
        const auto *node = r.try_get<SceneNode>(child);
        if (!node || node->Parent == entt::null) return I4;
        return ToMatrix(r.get<WorldTransform>(node->Parent)) * r.get<ParentInverse>(child).M;
    };
    static const auto LocalToMatrix = [](Transform &&t) { return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S); };

    const auto *node = r.try_get<SceneNode>(e);
    const bool has_parent = node && node->Parent != entt::null;
    const auto &wt = has_parent ? r.emplace_or_replace<WorldTransform>(e, MakeWorldTransform(GetParentDelta(r, e) * LocalToMatrix(GetTransform(r, e)))) : r.emplace_or_replace<WorldTransform>(e, MakeWorldTransform(GetTransform(r, e)));
    UpdateModelBuffer(r, e, wt);
    for (const auto child : Children{&r, e}) UpdateWorldTransform(r, child);
}

ChildrenIterator &ChildrenIterator::operator++() {
    if (Current != entt::null) {
        if (const auto *node = R->try_get<SceneNode>(Current)) Current = node->NextSibling;
        else Current = entt::null;
    }
    return *this;
}
ChildrenIterator ChildrenIterator::operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
}

ChildrenIterator Children::begin() const {
    if (ParentEntity == entt::null) return {R, entt::null};
    const auto *node = R->try_get<SceneNode>(ParentEntity);
    return {R, node ? node->FirstChild : entt::null};
}
// If no parent, returns the provided entity.
entt::entity GetParentEntity(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;

    if (const auto *node = r.try_get<SceneNode>(e)) {
        return node->Parent == entt::null ? e : node->Parent;
    }
    return e;
}

void ClearParent(entt::registry &r, entt::entity child) {
    if (child == entt::null || !r.all_of<SceneNode>(child)) return;

    const auto &child_node = r.get<const SceneNode>(child);
    const auto parent = child_node.Parent;
    if (parent == entt::null) return;

    const auto next_sibling = child_node.NextSibling;
    if (const auto &parent_node = r.get<const SceneNode>(parent);
        parent_node.FirstChild == child) {
        r.patch<SceneNode>(parent, [next_sibling](auto &n) { n.FirstChild = next_sibling; });
    } else {
        for (const auto sibling : Children(&r, parent)) {
            if (r.get<const SceneNode>(sibling).NextSibling == child) {
                r.patch<SceneNode>(sibling, [next_sibling](auto &n) { n.NextSibling = next_sibling; });
                break;
            }
        }
    }

    r.patch<SceneNode>(child, [](auto &n) {
        n.Parent = entt::null;
        n.NextSibling = entt::null;
    });
    r.remove<ParentInverse>(child);
    UpdateWorldTransform(r, child);
}

void SetParent(entt::registry &r, entt::entity child, entt::entity parent) {
    if (child == entt::null || parent == entt::null || child == parent) return;

    if (!r.all_of<SceneNode>(child)) r.emplace<SceneNode>(child);
    if (!r.all_of<SceneNode>(parent)) r.emplace<SceneNode>(parent);

    ClearParent(r, child);

    const auto first_child = r.get<const SceneNode>(parent).FirstChild;
    r.patch<SceneNode>(child, [parent, first_child](auto &n) {
        n.Parent = parent;
        n.NextSibling = first_child; // Add to parent's children list (at the beginning for simplicity)
    });
    r.patch<SceneNode>(parent, [child](auto &n) { n.FirstChild = child; });
    r.emplace_or_replace<ParentInverse>(child, glm::inverse(ToMatrix(r.get<WorldTransform>(parent))));
}
