#include "scene/SceneGraph.h"
#include "TransformMath.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"

mat4 GetParentDelta(const state::Scene &r, state::Entity e) {
    const auto *node = r.try_get<SceneNode>(e);
    if (!node || node->Parent == state::Null) return I4;
    return ToMatrix(r.get<WorldTransform>(node->Parent)) * r.get<ParentInverse>(e).M;
}

ChildrenIterator &ChildrenIterator::operator++() {
    if (Current != state::Null) {
        if (const auto *node = R->try_get<SceneNode>(Current)) Current = node->NextSibling;
        else Current = state::Null;
    }
    return *this;
}

ChildrenIterator Children::begin() const {
    if (ParentEntity == state::Null) return {R, state::Null};
    const auto *node = R->try_get<SceneNode>(ParentEntity);
    return {R, node ? node->FirstChild : state::Null};
}

state::Entity GetParentEntity(const state::Scene &r, state::Entity e) {
    if (e == state::Null) return state::Null;

    if (const auto *node = r.try_get<SceneNode>(e)) {
        return node->Parent == state::Null ? e : node->Parent;
    }
    return e;
}

state::Entity ParentOrNull(const state::Scene &r, state::Entity e) {
    const auto p = GetParentEntity(r, e);
    return p == e ? state::Null : p;
}

void ClearParent(state::Scene &r, state::Entity child) {
    if (child == state::Null || !r.all_of<SceneNode>(child)) return;

    const auto &child_node = r.get<const SceneNode>(child);
    const auto parent = child_node.Parent;
    if (parent == state::Null) return;

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
        n.Parent = state::Null;
        n.NextSibling = state::Null;
    });
    r.remove<ParentInverse>(child);
}

namespace {
void LinkChildToParent(state::Scene &r, state::Entity child, state::Entity parent) {
    if (!r.all_of<SceneNode>(child)) r.emplace<SceneNode>(child);
    if (!r.all_of<SceneNode>(parent)) r.emplace<SceneNode>(parent);

    ClearParent(r, child);

    const auto first_child = r.get<const SceneNode>(parent).FirstChild;
    r.patch<SceneNode>(child, [parent, first_child](auto &n) {
        n.Parent = parent;
        n.NextSibling = first_child;
    });
    r.patch<SceneNode>(parent, [child](auto &n) { n.FirstChild = child; });
}
} // namespace

void EnsureWorldTransform(state::Scene &r, state::Entity e) {
    if (r.all_of<WorldTransform>(e)) return;
    const auto *t = r.try_get<const Transform>(e);
    if (!t) return;
    if (const auto *node = r.try_get<const SceneNode>(e); node && node->Parent != state::Null) EnsureWorldTransform(r, node->Parent);
    r.emplace<WorldTransform>(e, ToTransform(GetParentDelta(r, e) * ToMatrix(*t)));
}

void UpdateWorldTransformRecursive(state::Scene &r, state::Entity e) {
    const auto *t = r.try_get<const Transform>(e);
    if (!t) return;
    if (const auto *node = r.try_get<const SceneNode>(e); node && node->Parent != state::Null) EnsureWorldTransform(r, node->Parent);
    r.emplace_or_replace<WorldTransform>(e, ToTransform(GetParentDelta(r, e) * ToMatrix(*t)));
    for (const auto child : Children{&r, e}) UpdateWorldTransformRecursive(r, child);
}

void BuildMissingWorldTransforms(state::Scene &r) {
    std::vector<state::Entity> missing;
    for (const auto e : r.view<const Transform>(state::Exclude<WorldTransform>)) missing.push_back(e);
    for (const auto e : missing) EnsureWorldTransform(r, e);
}

void SetParent(state::Scene &r, state::Entity child, state::Entity parent) {
    if (child == state::Null || parent == state::Null || child == parent) return;
    LinkChildToParent(r, child, parent);
    r.emplace<ParentInverse>(child, I4);
    UpdateWorldTransformRecursive(r, child);
}

void SetParentKeepWorld(state::Scene &r, state::Entity child, state::Entity parent) {
    if (child == state::Null || parent == state::Null || child == parent) return;
    EnsureWorldTransform(r, child);
    EnsureWorldTransform(r, parent);
    const auto child_world = ToMatrix(r.get<const WorldTransform>(child));
    const auto parent_world_inv = numeric::Inverse(ToMatrix(r.get<const WorldTransform>(parent)));
    LinkChildToParent(r, child, parent);
    r.emplace<ParentInverse>(child, I4);
    r.emplace_or_replace<Transform>(child, ToTransform(parent_world_inv * child_world));
    UpdateWorldTransformRecursive(r, child);
}
