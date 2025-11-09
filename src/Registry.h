#pragma once

#include <entt/entity/entity.hpp>

#include <string>
#include <string_view>

struct Name {
    std::string Value;
};

// Invariants:
// * Zero or more entities can be Selected.
// * The (single) most recently Selected entity is Active.
// NOTE: An entity can be both Selected and Active, and can also be Active but not Selected.
struct Selected {};
struct Active {};

struct Visible {}; // Visible in the scene
struct Frozen {}; // Disable entity transform changes

struct RenderInstance {
    uint32_t BufferIndex{0}; // slot in GPU model instance buffer
};

struct SceneNode {
    entt::entity Parent{entt::null};
    entt::entity FirstChild{entt::null};
    entt::entity NextSibling{entt::null};
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
    entt::entity FirstChild;

    ChildrenIterator begin() const { return {R, FirstChild}; }
    ChildrenIterator end() const { return {R, entt::null}; }
};

std::string IdString(entt::entity);
std::string GetName(const entt::registry &, entt::entity); // Returns name if present, otherwise hex ID.
std::string CreateName(const entt::registry &, std::string_view prefix = "Entity");

entt::entity FindActiveEntity(const entt::registry &); // If no active entity, returns entt::null.
entt::entity GetParentEntity(const entt::registry &, entt::entity); // If no parent, returns the provided entity.
