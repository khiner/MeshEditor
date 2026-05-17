#pragma once

#include <filesystem>
#include <variant>

namespace action::project {
struct ClearMeshes {};

struct LoadGltf {
    std::filesystem::path Path;
};
struct SaveGltf {
    std::filesystem::path Path;
};
struct LoadRealImpact {
    std::filesystem::path Directory;
};

using Actions = std::variant<ClearMeshes>;
using FallibleActions = std::variant<LoadGltf, SaveGltf, LoadRealImpact>;
} // namespace action::project
