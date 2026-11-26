#pragma once

#include "action/Emit.h"
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
// A material whose node graph computes inputs exports with those inputs baked to BakeSize by BakeSize textures.
// The save reports each material, input or mesh whose inputs it cannot bake, which exports with the material's values.
struct SaveGltf {
    std::filesystem::path Path;
    uint32_t BakeSize{1024};
};
// Writes the material and its node graph as a MaterialX document.
struct SaveMaterialX {
    state::Entity Material;
    std::filesystem::path Path;
};
struct LoadRealImpact {
    std::filesystem::path Path;
};
// Record the initial document state from an empty scene so projects created without an imported scene can replay.
struct LoadEmptyScene {};

using Action = std::variant<LoadDefaultScene, Load, LoadGltf, SaveGltf, LoadRealImpact, LoadEmptyScene, SaveMaterialX>;

// Handlers run GPU work synchronously; failures are reported through the registry's action::Errors sink.
void Apply(state::Registry &, const Action &);
} // namespace action::io
namespace action {
void Append(Destination, io::Action);
}
