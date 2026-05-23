#pragma once

#include <cstdint>
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
struct SetStudioEnvironment {
    uint32_t Index;
};
struct SetSourceIblIntensity {
    float Intensity;
};

using Actions = std::variant<NewDefaultScene, SetStudioEnvironment, SetSourceIblIntensity>;
using FallibleActions = std::variant<LoadGltf, SaveGltf, LoadRealImpact>;
} // namespace action::project
