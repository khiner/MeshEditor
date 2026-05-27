#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct MaterialStore {
    std::vector<std::string> Names;
};

struct MaterialDirty {
    uint32_t Index{0};
};
struct MeshMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
};
struct MeshMaterialSlotSelection {
    uint32_t PrimitiveIndex{0};
};

// Present iff a loaded glTF declared variants.
// Empty Active means no variant active - each primitive shows its source-default material
// (per spec, also applied per-primitive when the active variant has no mapping).
struct MaterialVariants {
    std::vector<std::string> Names;
    std::optional<uint32_t> Active;
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + Punctual bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};
