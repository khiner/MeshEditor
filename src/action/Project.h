#pragma once

#include <filesystem>
#include <variant>

namespace action::project {
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
using FallibleActions = std::variant<LoadGltf, SaveGltf, LoadRealImpact>;
} // namespace action::project
