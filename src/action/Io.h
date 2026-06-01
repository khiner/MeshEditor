#pragma once

#include <entt/entity/fwd.hpp>

#include <string>
#include <variant>

// Scene/document lifecycle: new scene plus the file-IO actions that load and save it.
// Paths are std::string (not fs::path) so the actions serialize natively into the log.
namespace action::io {
struct NewDefaultScene {};

struct LoadGltf {
    std::string Path;
};
struct SaveGltf {
    std::string Path;
};
struct LoadRealImpact {
    std::string Directory;
};

using Actions = std::variant<NewDefaultScene, LoadGltf, SaveGltf, LoadRealImpact>;
using Action = Actions;

// Handlers run GPU work synchronously; failures are reported through the registry's action::Errors sink.
void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::io
