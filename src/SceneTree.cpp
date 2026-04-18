#include "SceneTree.h"
#include "TransformMath.h"

#include <entt/entity/registry.hpp>

mat4 GetParentDelta(const entt::registry &r, entt::entity e) {
    const auto *node = r.try_get<SceneNode>(e);
    if (!node || node->Parent == entt::null) return I4;
    return ToMatrix(r.get<WorldTransform>(node->Parent)) * r.get<ParentInverse>(e).M;
}

ChildrenIterator &ChildrenIterator::operator++() {
    if (Current != entt::null) {
        if (const auto *node = R->try_get<SceneNode>(Current)) Current = node->NextSibling;
        else Current = entt::null;
    }
    return *this;
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
}

void SetParent(entt::registry &r, entt::entity child, entt::entity parent) {
    if (child == entt::null || parent == entt::null || child == parent) return;

    if (!r.all_of<SceneNode>(child)) r.emplace<SceneNode>(child);
    if (!r.all_of<SceneNode>(parent)) r.emplace<SceneNode>(parent);

    ClearParent(r, child);

    const auto first_child = r.get<const SceneNode>(parent).FirstChild;
    r.patch<SceneNode>(child, [parent, first_child](auto &n) {
        n.Parent = parent;
        n.NextSibling = first_child;
    });
    r.patch<SceneNode>(parent, [child](auto &n) { n.FirstChild = child; });

    // Default ParentInverse preserves child's world position under new parent.
    // Callers that want identity (e.g. bones for FK) overwrite after.
    r.emplace<ParentInverse>(child, glm::inverse(ToMatrix(r.get<WorldTransform>(parent))));
}
