#pragma once

#include <entt/entity/fwd.hpp>

#include <string>
#include <variant>

// Scene/document lifecycle: new scene plus the file-IO actions that load and save it.
// Paths are std::string (not fs::path) so the actions serialize natively into the log.
namespace action::io {
// Clear all scene content, leaving an empty scene.
struct Clear {};

struct LoadGltf {
    std::string Path;
};
struct SaveGltf {
    std::string Path;
};
struct LoadRealImpact {
    std::string Directory;
};

using Action = std::variant<Clear, LoadGltf, SaveGltf, LoadRealImpact>;

// Handlers run GPU work synchronously; failures are reported through the registry's action::Errors sink.
void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::io
