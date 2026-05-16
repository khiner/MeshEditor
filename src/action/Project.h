#pragma once

#include "entt_fwd.h"

#include <filesystem>

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

using Actions = entt::type_list<ClearMeshes>;
using FallibleActions = entt::type_list<LoadGltf, SaveGltf, LoadRealImpact>;
} // namespace action::project
