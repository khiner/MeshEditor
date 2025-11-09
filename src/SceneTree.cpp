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
