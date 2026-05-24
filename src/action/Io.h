#pragma once

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>

// Scene/document lifecycle: new scene plus the file-IO actions that load and save it.
namespace action::io {
struct NewDefaultScene {};

struct LoadGltf {
    std::filesystem::path Path;
};
struct SaveGltf {
    std::filesystem::path Path;
};
struct LoadRealImpact {
    std::filesystem::path Directory;
};

using Actions = std::variant<NewDefaultScene>;
using Action = Actions;
// File IO runs GPU work synchronously and reports failure inline.
using FallibleActions = std::variant<LoadGltf, SaveGltf, LoadRealImpact>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
std::expected<void, std::string> Apply(entt::registry &, entt::entity viewport, const FallibleActions &);
} // namespace action::io
