#include "SceneTree.h"

#include <entt/entity/registry.hpp>

#include <ranges>

namespace {
mat4 ToMatrix(Transform &&t) {
    return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
}

// How much the child's parent has transformed since parenting
mat4 GetParentDelta(entt::registry &r, entt::entity child) {
    const auto *node = r.try_get<SceneNode>(child);
    if (!node || node->Parent == entt::null) return I4;
    return r.get<WorldMatrix>(node->Parent).M * r.get<ParentInverse>(child).M;
}
} // namespace

Transform GetTransform(const entt::registry &r, entt::entity e) {
    return {r.get<Position>(e).Value, r.get<Rotation>(e).Value, r.all_of<Scale>(e) ? r.get<Scale>(e).Value : vec3{1}};
}

void UpdateWorldMatrix(entt::registry &r, entt::entity e) {
    const auto &world_matrix = r.emplace_or_replace<WorldMatrix>(e, GetParentDelta(r, e) * ToMatrix(GetTransform(r, e)));
    UpdateModelBuffer(r, e, world_matrix);
    for (const auto child : Children{&r, e}) UpdateWorldMatrix(r, child);
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

entt::entity GetParentEntity(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;

    if (const auto *node = r.try_get<SceneNode>(e)) {
        return node->Parent == entt::null ? e : node->Parent;
    }
    return e;
}

entt::entity GetRootEntity(const entt::registry &r, entt::entity e) {
    while (true) {
        const auto parent = GetParentEntity(r, e);
        if (parent == e) return e;
        e = parent;
    }
}

void SetParent(entt::registry &r, entt::entity child, entt::entity parent) {
    if (child == entt::null || parent == entt::null || child == parent) return;

    if (!r.all_of<SceneNode>(child)) r.emplace<SceneNode>(child);
    if (!r.all_of<SceneNode>(parent)) r.emplace<SceneNode>(parent);

    ClearParent(r, child);

    auto &child_node = r.get<SceneNode>(child);
    auto &parent_node = r.get<SceneNode>(parent);
    child_node.Parent = parent;
    // Add to parent's children list (at the beginning for simplicity)
    child_node.NextSibling = parent_node.FirstChild;
    parent_node.FirstChild = child;

    r.emplace_or_replace<ParentInverse>(child, glm::inverse(r.get<WorldMatrix>(parent).M));
}

void ClearParent(entt::registry &r, entt::entity child) {
    if (child == entt::null || !r.all_of<SceneNode>(child)) return;

    auto &child_node = r.get<SceneNode>(child);
    const auto parent = child_node.Parent;
    if (parent == entt::null) return;

    if (auto &parent_node = r.get<SceneNode>(parent);
        parent_node.FirstChild == child) {
        parent_node.FirstChild = child_node.NextSibling;
    } else {
        for (const auto sibling : Children(&r, parent)) {
            if (auto &sibling_node = r.get<SceneNode>(sibling);
                sibling_node.NextSibling == child) {
                sibling_node.NextSibling = child_node.NextSibling;
                break;
            }
        }
    }

    child_node.Parent = entt::null;
    child_node.NextSibling = entt::null;

    r.remove<ParentInverse>(child);
    UpdateWorldMatrix(r, child);
}
