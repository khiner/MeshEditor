#include "SceneTree.h"
#include "gpu/WorldTransform.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/matrix_decompose.hpp>

Transform GetTransform(const entt::registry &r, entt::entity e) {
    return {r.get<Position>(e).Value, r.get<Rotation>(e).Value, r.all_of<Scale>(e) ? r.get<Scale>(e).Value : vec3{1}};
}

static inline WorldTransform MakeWorldTransform(const Transform &t) { return {t.P, QuatToVec4(glm::normalize(t.R)), t.S}; }

static WorldTransform MakeWorldTransform(const mat4 &m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    glm::decompose(m, scale, rotation, translation, skew, perspective);
    return {translation, QuatToVec4(glm::normalize(rotation)), scale};
}

mat4 ToMatrix(const WorldTransform &wt) {
    return glm::translate(I4, wt.Position) * glm::mat4_cast(glm::normalize(Vec4ToQuat(wt.Rotation))) * glm::scale(I4, wt.Scale);
}

Transform MatrixToTransform(const mat4 &m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    if (!glm::decompose(m, scale, rotation, translation, skew, perspective)) return {};
    return {translation, glm::normalize(rotation), scale};
}

mat4 GetParentDelta(const entt::registry &r, entt::entity e) {
    const auto *node = r.try_get<SceneNode>(e);
    if (!node || node->Parent == entt::null) return I4;
    return ToMatrix(r.get<WorldTransform>(node->Parent)) * r.get<ParentInverse>(e).M;
}

void UpdateWorldTransform(entt::registry &r, entt::entity e, bool propagate_to_children) {
    static const auto TransformToMatrix = [](const Transform &t) {
        return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
    };

    const auto *node = r.try_get<SceneNode>(e);
    const bool has_parent = node && node->Parent != entt::null;
    const auto &wt = has_parent ? r.emplace_or_replace<WorldTransform>(e, MakeWorldTransform(GetParentDelta(r, e) * TransformToMatrix(GetTransform(r, e)))) : r.emplace_or_replace<WorldTransform>(e, MakeWorldTransform(GetTransform(r, e)));
    UpdateModelBuffer(r, e, wt);
    if (propagate_to_children) {
        for (const auto child : Children{&r, e}) UpdateWorldTransform(r, child);
    }
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
        n.NextSibling = first_child;
    });
    r.patch<SceneNode>(parent, [child](auto &n) { n.FirstChild = child; });
    r.emplace_or_replace<ParentInverse>(child, glm::inverse(ToMatrix(r.get<WorldTransform>(parent))));
}
