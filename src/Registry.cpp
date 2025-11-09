#include "Registry.h"

#include <entt/entity/registry.hpp>

#include <cstdint>
#include <format>
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

std::string IdString(entt::entity e) { return std::format("0x{:08x}", uint32_t(e)); }
std::string GetName(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(e)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(e);
}
std::string CreateName(const entt::registry &r, std::string_view prefix) {
    const std::string prefix_str{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        const auto view = r.view<const Name>();
        const auto name = i == 0 ? prefix_str : std::format("{}_{}", prefix, i);
        if (!std::ranges::any_of(view, [&](auto e) { return view.get<Name>(e).Value == name; })) {
            return name;
        }
    }
    assert(false);
    return prefix_str;
}

entt::entity FindActiveEntity(const entt::registry &registry) {
    auto all_active = registry.view<Active>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? entt::null : *all_active.begin();
}

entt::entity GetParentEntity(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return entt::null;

    if (const auto *node = r.try_get<SceneNode>(e)) {
        return node->Parent == entt::null ? e : GetParentEntity(r, node->Parent);
    }
    return e;
}

void LinkChild(entt::registry &r, entt::entity parent, entt::entity child) {
    auto &child_node = r.get_or_emplace<SceneNode>(child);
    child_node.Parent = parent;
    child_node.NextSibling = entt::null;

    auto &parent_node = r.get_or_emplace<SceneNode>(parent);
    if (parent_node.FirstChild == entt::null) {
        parent_node.FirstChild = child;
    } else {
        // Find last sibling and append
        auto last_sibling = parent_node.FirstChild;
        for (const auto sibling : Children{&r, parent}) last_sibling = sibling;
        r.get<SceneNode>(last_sibling).NextSibling = child;
    }
}

void UnlinkChild(entt::registry &r, entt::entity child) {
    auto *child_node = r.try_get<SceneNode>(child);
    if (!child_node || child_node->Parent == entt::null) return;

    auto &parent_node = r.get<SceneNode>(child_node->Parent);
    if (parent_node.FirstChild == child) {
        parent_node.FirstChild = child_node->NextSibling;
    } else {
        // Find previous sibling
        for (auto sibling : Children{&r, child_node->Parent}) {
            auto &sibling_node = r.get<SceneNode>(sibling);
            if (sibling_node.NextSibling == child) {
                sibling_node.NextSibling = child_node->NextSibling;
                break;
            }
        }
    }

    child_node->Parent = entt::null;
    child_node->NextSibling = entt::null;
}
