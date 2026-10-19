#pragma once

#include "state/Entity.h"

#include <filesystem>

namespace action::io {
struct LoadDefaultScene {};

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
// Record the initial document state from an empty scene so projects created without an imported scene can replay.
struct LoadEmptyScene {};

using Action = std::variant<LoadDefaultScene, Load, LoadGltf, SaveGltf, LoadRealImpact, LoadEmptyScene>;

// Handlers run GPU work synchronously; failures are reported through the registry's action::Errors sink.
void Apply(state::Registry &, const Action &);
} // namespace action::io
