#include "SceneTree.h"

#include <entt/entity/registry.hpp>

#include <ranges>

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

    // Ensure both entities have SceneNode components
    if (!r.all_of<SceneNode>(child)) r.emplace<SceneNode>(child);
    if (!r.all_of<SceneNode>(parent)) r.emplace<SceneNode>(parent);

    // Remove from current parent if any
    ClearParent(r, child);

    // Set new parent
    auto &child_node = r.get<SceneNode>(child);
    auto &parent_node = r.get<SceneNode>(parent);

    child_node.Parent = parent;

    // Add to parent's children list (at the beginning for simplicity)
    child_node.NextSibling = parent_node.FirstChild;
    parent_node.FirstChild = child;
}

void ClearParent(entt::registry &r, entt::entity child) {
    if (child == entt::null || !r.all_of<SceneNode>(child)) return;

    auto &child_node = r.get<SceneNode>(child);
    const auto parent = child_node.Parent;
    if (parent == entt::null) return;

    // Remove from parent's children list
    auto &parent_node = r.get<SceneNode>(parent);
    if (parent_node.FirstChild == child) {
        parent_node.FirstChild = child_node.NextSibling;
    } else {
        // Find the previous sibling
        for (auto sibling : Children(&r, parent)) {
            auto &sibling_node = r.get<SceneNode>(sibling);
            if (sibling_node.NextSibling == child) {
                sibling_node.NextSibling = child_node.NextSibling;
                break;
            }
        }
    }

    // Clear child's parent and sibling references
    child_node.Parent = entt::null;
    child_node.NextSibling = entt::null;
}
