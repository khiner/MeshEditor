#pragma once

#include "state/Entity.h"

#include <filesystem>
#include <variant>

// Scene/document lifecycle: new scene plus the file-IO actions that load and save it.
namespace action::io {
// Populate the default scene content.
struct LoadDefaultScene {};

// Load a file into the scene, choosing the importer by extension.
struct Load {
    std::filesystem::path Path;
};
struct LoadGltf {
    std::filesystem::path Path;
};
struct SaveGltf {
    std::filesystem::path Path;
};
struct LoadRealImpact {
    std::filesystem::path Path;
};

using Action = std::variant<LoadDefaultScene, Load, LoadGltf, SaveGltf, LoadRealImpact>;

// Handlers run GPU work synchronously; failures are reported through the registry's action::Errors sink.
void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::io
