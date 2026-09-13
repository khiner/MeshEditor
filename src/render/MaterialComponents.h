#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace store {
struct History;
}
namespace project {
template<typename T> struct VectorHistory;
}

struct MaterialStore {
    MaterialStore();
    ~MaterialStore();
    void AppendNames(std::vector<std::string>);
    void ResizeNames(size_t);
    void Track(store::History &);
    std::vector<std::string> Names;

private:
    std::unique_ptr<project::VectorHistory<std::string>> Tracked;
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

// Presence indicates declared glTF variants.
// An empty Active value selects each primitive's default material.
struct MaterialVariants {
    std::vector<std::string> Names;
    std::optional<uint32_t> Active;
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + Punctual bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};
